import numpy as np
import sys
import os

import utils
from config import DEBUG_MODE
from processing.process_obs import process_traces, compute_Fobs
from processing.process_aux import get_Fexp_and_mapping


def pairs_attack(obs, aux, exp_params):
    """
    Inference attack targeting temporal correlations in the SWAT/pancake trace.

    Targets a pair of tokens with the strongest observed co-occurrence within windows determined by the attack parameter 'latency',
    and returns the top-k candidate keywords and/or documents most likely to be their plaintext.

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
        target_token (int): Observed token id identified as the keyword side of the target pair.
        top_k_keys (list of int): Top-k plaintext keyword ids (in [0, nkw)) for target_token.
        target_doc_token (int): Observed token id identified as the document side of the target pair.
        doc_candidates (list of int): Top-k plaintext doc ids (in [nkw, nkw+ndoc)) for target_doc_token.
    """
    def_name = exp_params.def_params['name']
    if def_name == 'waffle':
        raise NotImplementedError("pairs_attack does not support waffle defense")

    nkw = exp_params.gen_params['nkw']
    ndoc = exp_params.gen_params['ndoc']
    N = nkw + ndoc
    top_k = exp_params.att_params['top_k']
    
    token_trace, token_info = process_traces(obs, aux, exp_params.def_params)
    print(f"token_trace: {token_trace[:20]}...", flush=True)
    latency = exp_params.att_params.get('latency', 1)
    nq_per_tok, Fobs = compute_Fobs(def_name, token_trace, len(token_info), latency)
    Fobs_counts = Fobs * nq_per_tok

    Fexp, rep_to_kw = get_Fexp_and_mapping(aux, exp_params.def_params, False)
    nrep = len(rep_to_kw)

    # 1) Target pair: the two observed tokens with strongest symmetric co-occurrence.
    cooc = Fobs_counts + Fobs_counts.T
    np.fill_diagonal(cooc, 0)
    t_a, t_b = np.unravel_index(np.argmax(cooc), cooc.shape)

    # 2) Real queries alternate kw -> doc. A kw token concentrates outgoing transitions
    #    on a few docs; a doc token spreads over many kws. Assign the more-concentrated
    #    outgoing distribution as the keyword side.
    if Fobs[:, t_a].max() >= Fobs[:, t_b].max():
        target_token, target_doc_token = int(t_a), int(t_b)
    else:
        target_token, target_doc_token = int(t_b), int(t_a)
    
    if DEBUG_MODE:
        print(f"rep_to_kw (len={len(rep_to_kw)}):\n{rep_to_kw}", flush=True)
        print(f"target_token: {target_token}, target_doc_token: {target_doc_token}", flush=True)
        dataset_name = exp_params.gen_params['dataset']
        seed = exp_params.gen_params.get('seed', -1)
        utils.plot_heatmap(cooc, "Token", f"{dataset_name}_{seed}_co_matrix.png")
    
    # 3) Collapse replica-level Fexp to plaintext-pair level via best-replica match.
    fexp_pp = np.zeros((N, N))
    for r_curr in range(nrep):
        p_curr = rep_to_kw[r_curr]
        if p_curr >= N: # Ignore dummy replicas
            continue  
        for r_next in range(nrep):
            p_next = rep_to_kw[r_next]
            if p_next >= N: # Ignore dummy replicas
                continue
            if Fexp[r_next, r_curr] > fexp_pp[p_next, p_curr]:
                fexp_pp[p_next, p_curr] = Fexp[r_next, r_curr]

    # 4) Score candidate (p_kw, p_doc) by the neg log-likelihood of ONLY the
    #    target pair's observed forward/backward transitions -- no marginalization
    #    over other tokens (unlike IHOP's full-assignment 'Freq' cost).
    n_fwd = Fobs_counts[target_doc_token, target_token]  # kw -> doc
    n_bwd = Fobs_counts[target_token, target_doc_token]  # doc -> kw
    log_fexp = np.log(np.maximum(fexp_pp, 1e-20))
    # cost shape (nkw, ndoc): cost[i, j] is for (p_kw=i, p_doc=nkw+j).
    cost = -n_fwd * log_fexp[nkw:, :nkw].T - n_bwd * log_fexp[:nkw, nkw:]

    # 5) Marginalize (min over the other side) to rank each side independently.
    top_k_keys = np.argsort(cost.min(axis=1))[:top_k].astype(int).tolist()
    doc_candidates = (np.argsort(cost.min(axis=0))[:top_k] + nkw).astype(int).tolist()

    return target_token, top_k_keys, target_doc_token, doc_candidates
