import os
import time
from experiment import run_experiment
import numpy as np
from exp_params import ExpParams


def print_exp_to_run(parameter_dict, n_runs):
    for key in parameter_dict:
        print('  {:s}: {}'.format(key, parameter_dict[key]))
    print("* Number of runs: {:d}".format(n_runs))


if __name__ == "__main__":

    os.system('mesg n')

    time_init = time.time()
    PRO_DATASETS_PATH = 'datasets_pro'
    EXPERIMENTS_PATH = 'results'

    exp_params = ExpParams()
    ###### Simple Enron experiment; accuracy for seed=0 should be 0.058, 0.174, 0.904, and 0.952 for niters=0, 10, 100, and 1000, respectively.
    exp_params.set_general_params(dataset='pages', nkw=100, nqr=10_000, freq='file',
                                  mode_ds='same', mode_fs='same', mode_kw='rand', mode_query='markov')
    exp_params.set_defense_params('none')
    attack_list = [('ihop', {'mode': 'Freq', 'niters': 1000, 'pfree': 0.25})]
    niter_list = [0, 10, 100, 1000]

    np.set_printoptions(precision=4)
    print(exp_params)
    acc_list = [[] for _ in attack_list]
    for seed in range(1):
        print("Seed: ", seed)

        for i_att, (att, att_p) in enumerate(attack_list):
            exp_params.set_attack_params(att, **att_p)
            exp_params.att_params['niter_list'] = niter_list
            acc, accu, time_exp = run_experiment(exp_params, seed=seed, debug_mode=True)
            if type(acc) == list:
                acc_list[i_att].append((acc[-1], accu[-1]))
                for acc, accu, niters in zip(acc, accu, exp_params.att_params['niter_list']):
                    print("{:d}-{:d}) {:s}, acc={:.3f}, accu={:.3f} ({:.2f} secs)".format(seed, niters, att, acc, accu, time_exp))
            else:
                acc_list[i_att].append((acc, accu))
                print("{:d}) {:s}, acc={:.3f}, accu={:.3f} ({:.2f} secs)".format(seed, att, acc, accu, time_exp))

    print("Summary of results:")
    for i_att, (att, att_p) in enumerate(attack_list):
        print("{:s}: avg acc={:.3f}, avg accu={:.3f}".format(att, *[np.mean(aux) for aux in zip(*acc_list[i_att])]))
