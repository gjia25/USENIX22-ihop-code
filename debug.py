import os
import time
from experiment import run_experiment
import numpy as np
from exp_params import ExpParams
from config import *

def print_exp_to_run(parameter_dict, n_runs):
    for key in parameter_dict:
        print('  {:s}: {}'.format(key, parameter_dict[key]))
    print("* Number of runs: {:d}".format(n_runs))

def print_log(string, file_handle):
    print(string)
    print(string, file=file_handle)

def run_experiment_wrapper(exp_params, attack_list):
    theta_tag = f"_th{exp_params.def_params['theta']}" if 'theta' in exp_params.def_params else ""
    log_dir = f"out/{exp_params.def_params['name']}_{exp_params.gen_params['dataset'][:5]}_{exp_params.def_params['name']}{theta_tag}_{CORR_LEVEL}_top{TOP_K_PAIRS}_nkw{exp_params.gen_params['nkw']}/"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # save config for this experiment
    with open(log_dir + '/config.txt', "w") as configOutFile:
        with open("config.py", "r") as conf:
            print(conf.read(), file=configOutFile)

    # save results for this experiment
    with open(log_dir + '/output.txt', "a") as f:
        np.set_printoptions(precision=4)
        print(exp_params, file=f)

        acc_list = [[] for _ in attack_list]
        for seed in range(NRUNS):
            print_log("Seed: {:d}".format(seed), f)

            for i_att, (att, att_p) in enumerate(attack_list):
                exp_params.set_attack_params(att, **att_p)
                acc, accu, time_exp = run_experiment(exp_params, seed=seed)
                if att == 'pairs':
                    # acc/accu are per-k hit indicators [hit@1, ..., hit@top_k] for kw/doc
                    top_k = att_p.get('top_k', TOP_K_PAIRS)
                    hits_str = "  ".join(f"top-{k}={acc[k-1]}" for k in range(1, top_k + 1))
                    doc_hits_str = "  ".join(f"top-{k}={accu[k-1]}" for k in range(1, top_k + 1))
                    print_log(f"{seed}) pairs (top_k={top_k}): kw: {hits_str}  doc: {doc_hits_str}  ({time_exp:.2f} secs)", f)
                    acc_list[i_att].append((acc, accu))
                elif type(acc) == list:
                    acc_list[i_att].append((acc[-1], accu[-1]))
                    for acc_, accu_ in zip(acc, accu):
                        print_log("{:d}) {:s}, acc={:.3f}, accu={:.3f} ({:.2f} secs)".format(seed, att, acc_, accu_, time_exp), f)
                else:
                    acc_list[i_att].append((acc, accu))
                    print_log("{:d}) {:s}, acc={:.3f}, accu={:.3f} ({:.2f} secs)".format(seed, att, acc, accu, time_exp), f)

        print_log("Summary of results:", f)
        for i_att, (att, att_p) in enumerate(attack_list):
            if att == 'pairs':
                top_k = att_p.get('top_k', TOP_K_PAIRS)
                kw_hits, doc_hits = zip(*acc_list[i_att])
                avg_kw = np.mean(kw_hits, axis=0)    # shape (top_k,)
                avg_doc = np.mean(doc_hits, axis=0)  # shape (top_k,)
                hits_str = "  ".join(f"top-{k}={avg_kw[k-1]:.3f}" for k in range(1, top_k + 1))
                doc_hits_str = "  ".join(f"top-{k}={avg_doc[k-1]:.3f}" for k in range(1, top_k + 1))
                print_log(f"pairs (top_k={top_k}): kw: {hits_str}  doc: {doc_hits_str}", f)
            else:
                print_log("{:s}: avg acc={:.3f}, avg accu={:.3f}".format(att, *[np.mean(aux) for aux in zip(*acc_list[i_att])]), f)

if __name__ == "__main__":

    os.system('mesg n')

    time_init = time.time()

    # Run Pancake
    exp_params = ExpParams()
    exp_params.set_defense_params('pancake')
    exp_params.set_general_params(dataset='enron-full', nkw=NKW, ndoc=NKW, nqr=NQR, freq='file', mode_query='markov')
    attack_list = [('pairs', {'top_k': TOP_K_PAIRS, 'latency': 1})]
    run_experiment_wrapper(exp_params, attack_list)

    thetas = []        # Iterate through powers of 2 for theta
    t = 1
    while t < NKW // 2:
        thetas.append(t)
        t *= 2
    thetas.append(NKW // 2)
    # latency = max(thetas)
    # Run SWAT with different theta
    for theta in thetas:
        latency = theta + 1 # window = 3*(theta+1), matches SWAT pool size
        print_log(f"\n\nRunning experiment with SWAT theta={theta}...", open(os.devnull, 'w'))
        exp_params = ExpParams()
        exp_params.set_defense_params('swat', theta=theta, sampling_func='Exp')
        exp_params.set_general_params(dataset='enron-full', nkw=NKW, ndoc=NKW, nqr=NQR, freq='file', mode_query='markov')
        attack_list = [
            # ('ihop', {'mode': 'Freq', 'pfree': PFREE, 'niters': NITERS, 'latency': latency}),
            ('pairs', {'top_k': TOP_K_PAIRS, 'latency': latency}),
        ]
        run_experiment_wrapper(exp_params, attack_list)

