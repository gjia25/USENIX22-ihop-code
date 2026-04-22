import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from config import DEBUG_MODE


def pairs_attack(obs, aux, exp_params):
    """
    Keyword inference attack targeting temporal correlations in the SWAT/pancake trace.

    Targets a single plaintext key — the one whose Markov transition is strongest —
    and returns the top-k candidate tokens most likely to be its replica.

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
                'top_k' (int): Number of candidate tokens to return for the target key.
            .def_params (dict): Defense parameters (name='pancake', etc.).
            .gen_params (dict): General parameters including:
                'nkw' (int): Number of keywords.
                'ndoc' (int): Number of documents.
                'nqr' (int): Number of real queries issued by the client.

    Returns:
        target_key (int): The single plaintext key predicted as the attack target
            (the key with the strongest Markov correlation signal).
        candidates (list of int): Top-k token ids predicted to be the replica of
            target_key, ordered by confidence (best first). Length == top_k.
    """
    def_name = exp_params.def_params['name']
    if def_name == 'waffle':
        raise NotImplementedError("pairs_attack does not support waffle defense")

    nkw = exp_params.gen_params['nkw']
    ndoc = exp_params.gen_params['ndoc']
    N = nkw + ndoc
    top_k = exp_params.att_params['top_k']

    F = aux['frequencies']  # N x N Markov matrix

    # ====================================================================
    # Step 1: Choose the single best target key pair
    #
    # score[j] = steady_state[j] * max_i F[i,j]: joint probability that j
    # occurs and transitions sharply to its most likely successor. The key
    # with the highest score has the strongest, most detectable correlation.
    # ====================================================================

    steady_state = utils.get_steady_state(F)

    max_trans_prob = np.max(F, axis=0)    # shape (N,): max outgoing prob per token
    best_successor = np.argmax(F, axis=0) # shape (N,): most likely next token
    joint_score = steady_state[:N] * max_trans_prob  # shape (N,)

    target_key = int(np.argmax(joint_score))
    target_key_corr = int(best_successor[target_key])

    print(f"\n[pairs_attack] Step 1: single target key pair")
    print(f"  target_key={target_key}  freq={steady_state[target_key]:.4f}"
          f"  corr_key={target_key_corr}  trans_prob={max_trans_prob[target_key]:.4f}"
          f"  joint_score={joint_score[target_key]:.4f}")

    # ====================================================================
    # Step 2: Co-occurrence matrix over sliding windows
    #
    # In SWAt, each real query enters the pool and may
    # be delayed by up to LATENCY pool cycles before emission. Each cycle
    # emits batch_size=3 observations. So a real query and its Markov
    # successor always appear within LATENCY+1 batches, giving a window of
    # batch_size * (LATENCY + 1) observations.
    # ====================================================================

    latency = exp_params.att_params['latency']
    batch_size = 3  # pancake emits 3 observations per client query
    window_size = batch_size * (latency + 1)

    token_ids = np.array([t for (t, _) in obs['traces']], dtype=np.int32)
    T = len(token_ids)

    max_token_id = int(token_ids.max()) + 1
    CO = np.zeros((max_token_id, max_token_id), dtype=np.float64)

    T_valid = T - window_size + 1
    for i in range(window_size):
        for j in range(window_size):
            if i == j:
                continue
            t1s = token_ids[i: T_valid + i]
            t2s = token_ids[j: T_valid + j]
            mask = t1s != t2s
            np.add.at(CO, (t1s[mask], t2s[mask]), 1)

    token_freq = np.bincount(token_ids, minlength=max_token_id).astype(np.float64)
    outer_freq = np.outer(token_freq, token_freq)
    outer_freq[outer_freq == 0] = 1.0
    CO_norm = CO / outer_freq

    if DEBUG_MODE:
        dataset_name = exp_params.gen_params['dataset']
        seed = exp_params.gen_params.get('seed', -1)
        # utils.plot_bar(np.bincount(token_ids).astype(float), "Token", f"{dataset_name}_{seed}_token_freq.png")
        utils.plot_heatmap(CO_norm, "Token", f"{dataset_name}_{seed}_co_matrix.png")

    # ====================================================================
    # Step 3: Top-k candidate tokens for target_key
    #
    # Rank all observed token pairs by directed lift CO_norm[ti, tj].
    # For each pair, use per-replica frequency proximity to orient which
    # of ti/tj is the better candidate for target_key (vs target_key_corr).
    # Collect top_k unique candidates in confidence order.
    # ====================================================================

    _, _, replicas_per_kw = utils.compute_pancake_parameters(N, steady_state)
    per_replica_freq = steady_state[:N] / replicas_per_kw[:N]  # shape (N,)

    token_freq_norm = token_freq / token_freq.sum()

    nonzero_idx = np.argwhere(CO > 0)
    if len(nonzero_idx) > 0:
        lift_scores = CO_norm[nonzero_idx[:, 0], nonzero_idx[:, 1]]
        order = np.argsort(lift_scores)[::-1]
        sorted_pairs = nonzero_idx[order]
    else:
        sorted_pairs = np.empty((0, 2), dtype=int)

    freq_target = per_replica_freq[target_key]
    freq_corr = per_replica_freq[target_key_corr]

    candidates = []
    seen = set()

    print(f"\n[pairs_attack] Step 3: top-{top_k} candidates for target_key={target_key}")
    print(f"  {'rank':>4}  {'token':>7}  {'lift':>10}  {'freq_err_target':>15}  {'freq_err_corr':>13}")

    for row in sorted_pairs:
        ti, tj = int(row[0]), int(row[1])
        # Assign the token whose frequency is closer to target_key's replica freq
        err_ti = abs(token_freq_norm[ti] - freq_target)
        err_tj = abs(token_freq_norm[tj] - freq_target)
        candidate = ti if err_ti <= err_tj else tj
        if candidate not in seen:
            print(f"  {len(candidates)+1:4d}  {candidate:7d}  {CO_norm[ti, tj]:10.4f}"
                  f"  {min(err_ti, err_tj):15.6f}  {abs(token_freq_norm[candidate] - freq_corr):13.6f}")
            candidates.append(candidate)
            seen.add(candidate)
        if len(candidates) == top_k:
            break

    # Pad with highest-frequency unseen tokens if fewer than top_k pairs were observed
    if len(candidates) < top_k:
        ranked_by_freq = np.argsort(np.abs(token_freq_norm - freq_target))
        for t in ranked_by_freq:
            if int(t) not in seen:
                candidates.append(int(t))
                seen.add(int(t))
            if len(candidates) == top_k:
                break

    return target_key, candidates
