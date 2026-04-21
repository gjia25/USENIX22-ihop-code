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
    log_dir = f"out/{exp_params.def_params['name']}_{exp_params.gen_params['dataset']}_{exp_params.def_params['name']}_th{THETA}_{CORR_LEVEL}_topk{TOP_K_PAIRS}_nkw{exp_params.gen_params['nkw']}/"
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
                acc, accu, time_exp = run_experiment(exp_params, seed=seed, debug_mode=True)
                if att == 'pairs':
                    # acc is a list of per-k hit indicators [hit@1, hit@2, ..., hit@top_k]
                    top_k = att_p.get('top_k', TOP_K_PAIRS)
                    hits_str = "  ".join(f"top-{k}={acc[k-1]}" for k in range(1, top_k + 1))
                    print_log(f"{seed}) pairs (top_k={top_k}): {hits_str}  ({time_exp:.2f} secs)", f)
                    acc_list[i_att].append(acc)
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
                avg_hits = np.mean(acc_list[i_att], axis=0)  # shape (top_k,)
                hits_str = "  ".join(f"top-{k}={avg_hits[k-1]:.3f}" for k in range(1, top_k + 1))
                print_log(f"pairs (top_k={top_k}): {hits_str}", f)
            else:
                print_log("{:s}: avg acc={:.3f}, avg accu={:.3f}".format(att, *[np.mean(aux) for aux in zip(*acc_list[i_att])]), f)

if __name__ == "__main__":

    os.system('mesg n')

    time_init = time.time()

    attack_list = [
        ('pairs', {'top_k': TOP_K_PAIRS}),
    ]

    # SWAT
    exp_params = ExpParams()
    exp_params.set_defense_params('swat')
    exp_params.set_general_params(dataset='enron-full', nkw=NKW, ndoc=NKW, nqr=NQR, freq='zipf', mode_ds='same', mode_fs='same', mode_kw='rand', mode_query='markov')
    run_experiment_wrapper(exp_params, attack_list)

