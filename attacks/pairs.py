import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from config import THETA


def pairs_attack(obs, aux, exp_params):
    """
    Keyword inference attack exploiting temporal correlations in the SWAT/pancake trace.

    Args:
        obs (dict): Observations produced by the defense. Keys:
            'traces' (list of tuple): One entry per observed query. For pancake
                (trace_type='tok_vol'), each entry is (token_id: int, volume: int=1),
                where token_id is a replica index in range(2 * N) with N = nkw + ndoc.
            'trace_type' (str): Format of traces, e.g. 'tok_vol' for pancake.
            'ndocs' (int): Number of documents in the client's dataset.

        aux (dict): Adversary's auxiliary information. Keys:
            'dataset' (list of list of int): Documents accessible to the adversary,
                each document is a list of keyword ids.
            'keywords' (range): Token ids available, range(nkw + ndoc).
            'frequencies' (np.ndarray): (N x N) Markov transition matrix.
                Column j = transition distribution from token j:
                frequencies[i, j] = P(next=i | current=j). Columns sum to 1.
            'mode_query' (str): Query generation mode, e.g. 'markov'.

        exp_params (ExpParams): Experiment parameters with fields:
            .att_params (dict): Attack hyperparameters:
                'num_targets' (int): Number of target key pairs to identify (default 50).
            .def_params (dict): Defense parameters (name='pancake', etc.).
            .gen_params (dict): General parameters including:
                'nkw' (int): Number of keywords.
                'ndoc' (int): Number of documents.
                'nqr' (int): Number of real queries issued by the client.

    Returns:
        list of int: Predicted token id for each entry in obs['traces'],
            length == len(obs['traces']). Each value is in range(N+1)
            where N = nkw + ndoc (N itself is the pancake dummy key index).
    """
    def_name = exp_params.def_params['name']
    if def_name == 'waffle':
        raise NotImplementedError("pairs_attack does not support waffle defense")

    nkw = exp_params.gen_params['nkw']
    ndoc = exp_params.gen_params['ndoc']
    N = nkw + ndoc
    num_targets = exp_params.att_params['num_targets']

    F = aux['frequencies']  # N x N Markov matrix

    # ====================================================================
    # Step 1: Choose target key pairs (strongest direct one-hop correlation)
    #
    # For each token j, score[j] = steady_state[j] * max_i F[i,j] is the
    # joint probability of j occurring and transitioning to its most likely
    # successor. This favours tokens that are both frequent and peaked,
    # since a rare token with a sharp transition yields few observable pairs.
    # ====================================================================

    steady_state = utils.get_steady_state(F)

    max_trans_prob = np.max(F, axis=0)    # shape (N,): max outgoing prob per token
    best_successor = np.argmax(F, axis=0) # shape (N,): most likely next token
    joint_score = steady_state[:N] * max_trans_prob  # shape (N,): freq * trans_prob

    target_key_indices = np.argsort(joint_score)[::-1][:num_targets]

    num_to_print = min(num_targets, 10)
    print(f"\n[pairs_attack] Step 1: top {num_to_print} target keys (direct one-hop, sorted by joint prob)")
    print(f"  {'key':>6}  {'freq':>8}  {'corr_key':>8}  {'trans_prob':>10}  {'joint_score':>11}")
    for k in target_key_indices[:num_to_print]:
        k_corr = int(best_successor[k])
        print(f"  {int(k):6d}  {steady_state[k]:8.4f}  {k_corr:8d}  {max_trans_prob[k]:10.4f}  {joint_score[k]:11.4f}")

    # ====================================================================
    # Step 2: Co-occurrence matrix over sliding windows
    #
    # In pancake + theta-decorr, each real query enters the pool and may
    # be delayed by up to THETA pool cycles before emission. Each cycle
    # emits batch_size=3 observations. So a real query and its Markov
    # successor always appear within THETA+1 batches, giving a window of
    # batch_size * (THETA + 1) observations.
    # ====================================================================

    batch_size = 3  # pancake emits 3 observations per client query
    window_size = batch_size * (THETA + 1)

    token_ids = np.array([t for (t, _) in obs['traces']], dtype=np.int32)
    T = len(token_ids)

    max_token_id = int(token_ids.max()) + 1
    CO = np.zeros((max_token_id, max_token_id), dtype=np.float64)

    # For each ordered offset pair (i, j) within the window, accumulate
    # CO[token_ids[k+i], token_ids[k+j]] over all valid positions k.
    T_valid = T - window_size + 1
    for i in range(window_size):
        for j in range(window_size):
            if i == j:
                continue
            t1s = token_ids[i: T_valid + i]
            t2s = token_ids[j: T_valid + j]
            mask = t1s != t2s  # skip self-pairs
            np.add.at(CO, (t1s[mask], t2s[mask]), 1)

    # Lift = CO / E[CO under independence]; removes bias from frequent tokens.
    token_freq = np.bincount(token_ids, minlength=max_token_id).astype(np.float64)
    outer_freq = np.outer(token_freq, token_freq)
    outer_freq[outer_freq == 0] = 1.0  # avoid divide-by-zero for unseen tokens
    CO_norm = CO / outer_freq

    # ====================================================================
    # Step 3: Greedy matching of token_ids to target keys
    #
    # Target pairs are ranked by Markov transition probability (highest =
    # most predictable). Observed token_id pairs are ranked by lift. We
    # greedily assign the top unassigned observed pair to the top unassigned
    # target pair, using transition frequency to define priority.
    # ====================================================================

    target_pairs = sorted(
        [(int(k), int(best_successor[k]), float(max_trans_prob[k])) for k in target_key_indices],
        key=lambda x: x[2], reverse=True
    )

    nonzero_idx = np.argwhere(CO > 0)
    if len(nonzero_idx) > 0:
        lift_scores = CO_norm[nonzero_idx[:, 0], nonzero_idx[:, 1]]
        order = np.argsort(lift_scores)[::-1]
        sorted_pairs = nonzero_idx[order]  # shape (M, 2), sorted by lift descending
    else:
        sorted_pairs = np.empty((0, 2), dtype=int)

    token_to_key = {}
    assigned_tokens = set()
    assigned_keys = set()

    print(f"\n[pairs_attack] Step 3: greedy token_id → key mapping")
    print(f"  {'token_i':>7} → {'key_i':>6}  |  {'token_j':>7} → {'key_j':>6}  |  {'CO':>10}  {'lift':>10}")
    for k, k_corr, _ in target_pairs:
        if k in assigned_keys or k_corr in assigned_keys:
            continue
        for row in sorted_pairs:
            ti, tj = int(row[0]), int(row[1])
            if ti not in assigned_tokens and tj not in assigned_tokens and ti != tj:
                token_to_key[ti] = k
                token_to_key[tj] = k_corr
                assigned_tokens.add(ti)
                assigned_tokens.add(tj)
                assigned_keys.add(k)
                assigned_keys.add(k_corr)
                print(f"  {ti:7d} → {k:6d}  |  {tj:7d} → {k_corr:6d}  |  {CO[ti, tj]:10.0f}  {CO_norm[ti, tj]:10.4f}")
                break

    # ====================================================================
    # Step 4: Predict for every trace entry
    #
    # Use the step-3 mapping for identified token_ids. For all others,
    # use linear_sum_assignment over unassigned_tokens x unassigned_keys
    # so each unassigned key is claimed by exactly one token. Any leftover
    # tokens (beyond the number of unassigned keys) use plain argmin.
    # ====================================================================
    from scipy.optimize import linear_sum_assignment

    _, _, replicas_per_kw = utils.compute_pancake_parameters(N, steady_state)
    per_replica_freq = steady_state[:N] / replicas_per_kw[:N]  # shape (N,)

    token_freq_norm = token_freq / token_freq.sum()

    unassigned_tokens = [t for t in range(max_token_id) if t not in assigned_tokens]
    unassigned_keys = [k for k in range(N) if k not in assigned_keys]

    fallback_map = {}
    if unassigned_keys:
        cost = np.abs(
            token_freq_norm[unassigned_tokens, np.newaxis]
            - per_replica_freq[unassigned_keys][np.newaxis, :]
        )  # shape (len(unassigned_tokens), len(unassigned_keys))
        cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_token_set = set()
        for r, c in zip(row_ind, col_ind):
            fallback_map[unassigned_tokens[r]] = unassigned_keys[c]
            matched_token_set.add(unassigned_tokens[r])
        # leftover tokens: not enough unassigned keys to cover all — use argmin
        for t in unassigned_tokens:
            if t not in matched_token_set:
                fallback_map[t] = int(np.argmin(np.abs(token_freq_norm[t] - per_replica_freq)))
    else:
        for t in unassigned_tokens:
            fallback_map[t] = int(np.argmin(np.abs(token_freq_norm[t] - per_replica_freq)))

    all_token_to_key = {**{t: token_to_key[t] for t in assigned_tokens}, **fallback_map}
    predictions = [all_token_to_key[int(t)] for (t, _) in obs['traces']]
    return predictions, all_token_to_key, token_to_key
