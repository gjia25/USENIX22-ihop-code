import os
import numpy as np
import pickle
import time
import attacks
from matplotlib import pyplot as plt
import utils
from defense import generate_observations
from collections import Counter, defaultdict
from config import *

def load_pro_dataset(dataset_name):
    full_path = os.path.join(PRO_DATASET_FOLDER, dataset_name + '.pkl')
    if not os.path.exists(full_path):
        raise ValueError("The file {} does not exist".format(full_path))

    with open(full_path, "rb") as f:
        dataset, keywords, aux = pickle.load(f)

    return dataset, keywords, aux


def generate_keyword_queries(mode_query, frequencies, nqr, nkw):
    # mode_query should always be 'markov' for the scenarios we consider.
    if mode_query == 'markov':
        assert frequencies.ndim == 2

        ## TRANSCRIPT GENERATION
        queries = []

        while len(queries) < nqr:
            # pick a keyword
            kwQuery = np.random.choice(list(range(nkw)), p=frequencies[:nkw, nkw])

            # pick a document based on the keyword
            docQuery = np.random.choice(list(range(nkw, len(frequencies))), p=frequencies[nkw:, kwQuery]) # for a skewed distribution, include that as p here

            queries.append(kwQuery)
            queries.append(docQuery)

    else:
        raise ValueError("Frequencies has {:d} dimensions, only 1 or 2 allowed".format(frequencies.ndim))
    return queries

def build_frequencies_from_file(chosen_kw_indices, chosen_doc_indices, dataset, trends):
    num_keys = len(chosen_kw_indices) + len(chosen_doc_indices)
    freq_real = np.zeros((num_keys, num_keys))

    # filter trends to chosen kws, collapse 52 weeks of trend data to a 1d array
    trend_matrix = trends[chosen_kw_indices, :]
    for i_col in range(trend_matrix.shape[1]):
        if sum(trend_matrix[:, i_col]) == 0:
            print("The {:d}th column of the trend matrix adds up to zero, making it uniform!".format(i_col))
            trend_matrix[:, i_col] = 1 / len(chosen_kw_indices)
        else:
            trend_matrix[:, i_col] = trend_matrix[:, i_col] / sum(trend_matrix[:, i_col])
    kw_freq = np.mean(trend_matrix, axis=1)

    # build transitions from docs to kws. No matter which doc, probability vector of transiting to any kw is kw_freq
    for doc_idx in range(len(chosen_kw_indices), num_keys):
        freq_real[0:len(chosen_kw_indices), doc_idx] = kw_freq

    # build transitions from kws to docs
    if CORR_LEVEL == 'high':
        # transition probability from kw to first doc containing it is 1, else 0
        for kw_idx, kw in enumerate(chosen_kw_indices):
            doc_selected_for_kw = False

            # select random doc containing this kw if HIGH_CORR_PERMUTE, else select first doc containing it
            docs = np.random.permutation(list(enumerate(chosen_doc_indices))) if HIGH_CORR_PERMUTE else enumerate(chosen_doc_indices)

            for doc_i, doc_n in docs:
                if doc_selected_for_kw: continue

                doc = dataset[doc_n]
                doc_idx = doc_i + len(chosen_kw_indices)
                if kw in doc:
                    freq_real[doc_idx, kw_idx] = 1
                    doc_selected_for_kw = True
                    if kw_idx == 0: print("doc for kw0 is", doc_idx)
    else:
        # transition probability from kw for n docs containing it is default 1/n each
        exp_factor = 1.0

        # possibly weight later indices higher
        if CORR_LEVEL == 'mid':
            exp_factor = 1.2

        for kw_idx, kw in enumerate(chosen_kw_indices):
            weight = 1
            totalOfWeights = 0
            for doc_i, doc_n in enumerate(chosen_doc_indices):
                doc = dataset[doc_n]
                doc_idx = doc_i + len(chosen_kw_indices)
                if kw in doc:
                    weight *= exp_factor
                    freq_real[doc_idx, kw_idx] = weight
                    totalOfWeights += weight
            # normalize so probabilities for each kw sum to 1
            for doc_i, doc in enumerate(chosen_doc_indices):
                doc_idx = doc_i + len(chosen_kw_indices)
                freq_real[doc_idx, kw_idx] /= totalOfWeights

    # sanity check
    # column i = probability vector of transitioning from token i to other tokens (IHOP appendix D)
    # sum of column i should be 1
    import math
    for r in range(num_keys):
        assert(math.isclose(sum(freq_real[:, r]), 1))

    return freq_real, freq_real, freq_real

def generate_train_test_data(gen_params):
    nkw = gen_params['nkw'] # number of keywords in the datastore
    ndoc = gen_params['ndoc'] # nubmer of documents in the datastore
    dataset_name = gen_params['dataset']
    mode_kw = gen_params['mode_kw']
    mode_ds = gen_params['mode_ds']
    freq_name = gen_params['freq']
    mode_fs = gen_params['mode_fs']

    # Load the dataset for this experiment
    dataset, keywords, aux_dataset_info = load_pro_dataset(dataset_name)

    # use top strategy to pick keywords - picking most popular minimizes likelihood that some kw appears in no doc
    chosen_kw_indices = list(range(nkw))
    # rand strategy: chosen_kw_indices = np.random.permutation(len(keywords))

    # use rand strategy to pick docs - could also do with top
    permutation = np.random.permutation(len(dataset))
    chosen_doc_indices = list(permutation[:ndoc])

    # get Markov transition matrix
    freq_adv, freq_cli, freq_real = build_frequencies_from_file(chosen_kw_indices, chosen_doc_indices, dataset, aux_dataset_info['trends'])

    full_data_adv = {'dataset': dataset,
                     'keywords': range(nkw+ndoc),
                     'frequencies': freq_adv,
                     'mode_query': gen_params['mode_query']}
    full_data_client = {'dataset': dataset,
                        'keywords': range(nkw+ndoc),
                        'frequencies': freq_cli}
    return full_data_adv, full_data_client, freq_real


def run_attack(attack_name, **kwargs):
    if attack_name == 'freq':
        return attacks.freq_attack(**kwargs)
    elif attack_name == 'sap':
        return attacks.sap_attack(**kwargs)
    elif attack_name == 'ihop':
        return attacks.ihop_attack(**kwargs)
    elif attack_name == 'umemaya':
        return attacks.umemaya_attack(**kwargs)
    elif attack_name == 'fastpfp':
        return attacks.fastfpf_attack(**kwargs)
    elif attack_name == 'ikk':
        return attacks.ikk_attack(**kwargs)
    elif attack_name == 'graphm':
        return attacks.graphm_attack(**kwargs)
    else:
        raise ValueError("Attack name '{:s}' not recognized".format(attack_name))


def run_experiment(exp_param, seed, debug_mode=False):
    v_print = print if debug_mode else lambda *a, **k: None

    t0 = time.time()
    np.random.seed(seed)
    full_data_adv, full_data_client, freq_real = generate_train_test_data(exp_param.gen_params)
    v_print("Generated train-test data: adv dataset {:d}, client dataset {:d} ({:.1f} secs)".format(len(full_data_adv['dataset']),
                                                                                                    len(full_data_client['dataset']),
                                                                                                    time.time() - t0))

    real_queries = generate_keyword_queries(exp_param.gen_params['mode_query'], freq_real, exp_param.gen_params['nqr'])
    v_print("Generated {:d} real queries ({:.1f} secs)".format(len(real_queries), time.time() - t0))

    observations, bw_overhead, real_and_dummy_queries = generate_observations(full_data_client, exp_param.def_params, real_queries)
    v_print("Applied defense ({:.1f} secs)".format(time.time() - t0))

    keyword_predictions_for_each_query = run_attack(exp_param.att_params['name'], obs=observations, aux=full_data_adv, exp_params=exp_param)
    v_print("Done running attack ({:.1f} secs)".format(time.time() - t0))
    time_exp = time.time() - t0

    # Compute accuracy
    if type(keyword_predictions_for_each_query) == list and type(keyword_predictions_for_each_query[0]) != list:
        acc_vector = np.array([1 if query == prediction else 0 for query, prediction in zip(real_and_dummy_queries, keyword_predictions_for_each_query)])
        acc_un_vector = np.array([np.mean(acc_vector[real_and_dummy_queries == i]) for i in set(real_and_dummy_queries)])
        accuracy = np.mean(acc_vector)
        accuracy_un = np.mean(acc_un_vector)
        return accuracy, accuracy_un, time_exp
    elif type(keyword_predictions_for_each_query) == list and type(keyword_predictions_for_each_query[0]) == list:
        acc_list, acc_un_list = [], []
        for pred in keyword_predictions_for_each_query:
            acc_vector = np.array([1 if query == prediction else 0 for query, prediction in zip(real_and_dummy_queries, pred)])
            # print(np.mean(np.array([1 if query == prediction else 0 for query, prediction in zip(real_and_dummy_queries, pred)])), np.mean(np.array([1 if real == prediction else 0 for real, prediction in zip(real_and_dummy_queries, pred)])))
            acc_un_vector = np.array([np.mean(acc_vector[real_and_dummy_queries == i]) for i in set(real_and_dummy_queries)])
            acc_list.append(np.mean(acc_vector))
            acc_un_list.append(np.mean(acc_un_vector))
        return acc_list, acc_un_list, time_exp
    else:
        return -1, -1, -1
