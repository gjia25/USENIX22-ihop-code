import os
import numpy as np
import pandas as pd
import pickle
import time
import attacks
from matplotlib import pyplot as plt
import utils
from defense import generate_observations
from collections import Counter
from config import PRO_DATASET_FOLDER


def load_pro_dataset(dataset_name):
    full_path = os.path.join(PRO_DATASET_FOLDER, dataset_name + '.pkl')
    if not os.path.exists(full_path):
        raise ValueError("The file {} does not exist".format(full_path))

    with open(full_path, "rb") as f:
        dataset, keywords, aux = pickle.load(f)

    return dataset, keywords, aux


def generate_keyword_queries(mode_query, frequencies, nqr):
    nkw = frequencies.shape[0]
    if mode_query == 'iid':
        assert frequencies.ndim == 1
        queries = list(np.random.choice(list(range(nkw)), nqr, p=frequencies))
    elif mode_query == 'markov':
        assert frequencies.ndim == 2
        ss = utils.get_steady_state(frequencies)
        queries = np.zeros(nqr, dtype=int)
        queries[0] = np.random.choice(len(range(nkw)), p=ss)
        for i in range(1, nqr):
            queries[i] = np.random.choice(len(range(nkw)), p=frequencies[:, queries[i - 1]])
    elif mode_query == 'each':
        queries = list(np.random.permutation(nkw))[:min(nqr, nkw)]
    else:
        raise ValueError("Frequencies has {:d} dimensions, only 1 or 2 allowed".format(frequencies.ndim))
    return queries


def build_frequencies_from_file(dataset_name, chosen_kw_indices, keywords, aux_dataset_info, mode_fs):
    def _process_markov_matrix(months):

        m = np.zeros((nkw, nkw))
        msink = np.zeros(nkw)
        for month in months:
            m += aux_dataset_info['transitions'][month][np.ix_(chosen_kw_indices, chosen_kw_indices)]
            msink += aux_dataset_info['transitions'][month][chosen_kw_indices, -1]

        sink_profile = msink / np.sum(msink)
        cols_sunk = [val[0] for val in np.argwhere(m.sum(axis=0) == 0)]
        m[:, cols_sunk] = np.ones(len(cols_sunk)) * msink.reshape(nkw, 1)

        # Add a certain probability of restart
        m = m / m.sum(axis=0)
        p_restart = 0.05  # DONE!
        m_new = (1 - p_restart) * m + p_restart * sink_profile.reshape(len(sink_profile), 1)

        # m_markov = m_new / m_new.sum(axis=0)
        ss = utils.get_steady_state(m_new)
        if any(ss < -1e-8):
            print(ss[ss < 0])
        return m_new

    nkw = len(chosen_kw_indices)
    if dataset_name in ('enron-full', 'lucene', 'bow-nytimes', 'articles1', 'movie-plots'):
        trend_matrix = aux_dataset_info['trends'][chosen_kw_indices, :]
        for i_col in range(trend_matrix.shape[1]):
            if sum(trend_matrix[:, i_col]) == 0:
                print("The {:d}th column of the trend matrix adds up to zero, making it uniform!".format(i_col))
                trend_matrix[:, i_col] = 1 / nkw
            else:
                trend_matrix[:, i_col] = trend_matrix[:, i_col] / sum(trend_matrix[:, i_col])
        if mode_fs == 'same':  # Take last year of data
            freq_cli = freq_real = freq_adv = np.mean(trend_matrix, axis=1)
        elif mode_fs == 'past':  # First half of year for adv, last half is real and client's
            freq_adv = np.mean(trend_matrix[:, -52:-26], axis=1)
            freq_cli = freq_real = np.mean(trend_matrix[:, -26:], axis=1)
        else:
            raise ValueError("Frequencies split mode '{:s}' not allowed for {:s}".format(mode_fs, dataset_name))
    elif dataset_name.startswith('wiki'):
        # category = dataset_name[5:]
        if mode_fs == 'same':
            months_real = range(7, 13)
            months_adv = range(7, 13)
        elif mode_fs == 'past':
            months_real = range(7, 13)
            months_adv = range(1, 7)
        elif mode_fs == 'same1':  # December vs December
            months_real = [12]
            months_adv = [12]
        elif mode_fs == 'past1':  # June vs December
            months_real = [12]
            months_adv = [6]
        else:
            raise ValueError("Frequencies split mode '{:s}' not allowed for {:s}".format(mode_fs, dataset_name))
        freq_adv = _process_markov_matrix(months_adv)
        freq_real = _process_markov_matrix(months_real)
        freq_cli = _process_markov_matrix(months_real)
    else:
        raise ValueError("No frequencies for dataset {:s}".format(dataset_name))
    return freq_adv, freq_cli, freq_real

def generate_page_observations(gen_params):
    assert gen_params['dataset'] == 'pages'
    data_path = "/Users/user/Desktop/Yale/Y2/CPSC581/final"
    num_features = 26
    input_cols = [f'page_{i+1}' for i in range(num_features)]
    target_cols = [f'idx_{i+1}' for i in range(num_features)]
    
    def _filter_by_unique(data, unique_indices):
        return data[data[target_cols].isin(unique_indices).all(axis=1)]
    def _get_data_adv(nkw):
        all_filename = f"{data_path}/all.csv"
        train_filename = f"{data_path}/train.csv"
        path_data_adv = f"{data_path}/data_adv.pkl"
        
        inputs = pd.read_csv(all_filename)
        train_data = pd.read_csv(train_filename)
        # unique_pages = pd.unique(train_data[input_cols].values.ravel('K'))
        
        print("Loaded datasets.")
        if os.path.exists(path_data_adv):
            print(f"Loading auxiliary dataset...")
            with open(path_data_adv, 'rb') as f:
                data_adv = pickle.load(f)
        else:
            print(f"Building auxiliary dataset...")              
            # data_adv = [[train_data.iloc[row][col]] for row in range(len(train_data)) for col in target_cols]
            page_to_indices = {}
            for i, row in train_data.iterrows():
                for j in range(1,num_features+1):
                    page = row[f'page_{j}']
                    if page not in page_to_indices:
                        page_to_indices[page] = set()
                    page_to_indices[page].add(row[f'idx_{j}'])
                if i % 500000 == 0:
                    print(i, end=' ', flush=True)
            data_adv = list(map(lambda x: list(x), page_to_indices.values()))
            print(data_adv[:5]) # kws (entries) in each doc (page)
            with open(path_data_adv, 'wb') as f:
                pickle.dump(data_adv, f)
            print(f"aux written to {path_data_adv}")  

        unique_indices = pd.Series(inputs[target_cols].values.ravel('K')).value_counts().index.tolist()
        if nkw:
            unique_indices = unique_indices[:nkw]
            _filter_by_unique(train_data, unique_indices)
            print(f"len(train_data) = {len(train_data)}.")
        n = len(unique_indices)
        value_to_idx = {value: idx for idx, value in enumerate(unique_indices)}
        chosen_kw_indices = [range(n)]
        print(f"nkw = {n}. Building F_aux...")
        Faux = np.zeros((n, n))
        m = np.zeros((n, n))
        for i, row in train_data.iterrows():
            inference_request = row[target_cols].values.tolist()
            m += np.histogram2d(inference_request, inference_request, bins=(range(n + 1), range(n + 1)))[0]
            if i % 500000 == 0:
                print(i, end=' ', flush=True)
        for j in range(n):
            if np.sum(m[:, j]) > 0:
                Faux[:, j] = m[:, j] / np.sum(m[:, j])
        # for _, row in train_data.iterrows():
        #     indexes = [value_to_idx[row[col]] for col in target_cols]
        #     for i in indexes:
        #         for j in indexes:
        #             m[i, j] += 1
        # m = m / m.sum(axis=0)
        del inputs, train_data
        return data_adv, Faux, chosen_kw_indices, unique_indices, value_to_idx
    def _get_traces_ideal(test_data, value_to_idx):
        test_data[target_cols] = test_data[target_cols].map(lambda v: value_to_idx[v])
        traces = []
        for _, row in test_data.iterrows():
            for i in range(num_features):
                traces.append((f"{row[f'page_{i}']}_{row[f'idx_{i}']}", row[f'idx_{i}'])) # (doc_id, kw_id)
        return traces
    
    data_adv, Faux, chosen_kw_indices, unique_indices, value_to_idx = _get_data_adv(gen_params['nkw'])
    full_data_adv = {'dataset': data_adv,
                     'keywords': chosen_kw_indices,
                     'frequencies': Faux,
                     'mode_query': gen_params['mode_query']}
    print('Generated auxiliary dataset')
    test_filename = f"{data_path}/outfiles/kt2truthconv.csv"
    test_data = pd.read_csv(test_filename)
    if gen_params['nkw']:
        _filter_by_unique(test_data, unique_indices)
        assert len(test_data) > 0
    print(f"nqr = {len(test_data)}. Generating real accesses...")
    real_queries = test_data[target_cols].values.ravel().tolist()
    assert len(real_queries) == len(test_data) * num_features
    print('Generating trace...')
    observations = {}
    observations['trace_type'] = 'ap_unique'
    observations['traces'] = _get_traces_ideal(test_data, value_to_idx)
    observations['ndocs'] = len(data_adv)
    print('Done.')

    return full_data_adv, observations, real_queries

def generate_train_test_data(gen_params):
    nkw = gen_params['nkw']
    dataset_name = gen_params['dataset']
    mode_kw = gen_params['mode_kw']
    mode_ds = gen_params['mode_ds']
    freq_name = gen_params['freq']
    mode_fs = gen_params['mode_fs']

    # Load the dataset for this experiment
    dataset, keywords, aux_dataset_info = load_pro_dataset(dataset_name)
    ndoc = len(dataset) if gen_params['ndoc'] == 'full' else min(len(dataset), gen_params['ndoc'])

    # Select the keywords for this experiment
    if mode_kw == 'top':
        kw_counter = Counter([kw for document in dataset for kw in document])
        chosen_kw_indices = sorted(kw_counter.keys(), key=lambda x: kw_counter[x], reverse=True)[:nkw]
    elif mode_kw == 'rand':
        permutation = np.random.permutation(len(keywords))
        chosen_kw_indices = list(permutation[:nkw])
    else:
        raise ValueError("Keyword selection mode '{:s}' not allowed".format(mode_kw))

    # Get client dataset and adversary's auxiliary dataset
    dataset = [dataset[i] for i in np.random.permutation(len(dataset))[:ndoc]]
    if mode_ds.startswith('same'):
        percentage = 100 if mode_ds == 'same' else int(mode_ds[4:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[:int(len(dataset) * percentage / 100)]]
        data_adv = dataset_selection
        data_cli = dataset_selection
    elif mode_ds.startswith('common'):
        percentage = 50 if mode_ds == 'common' else int(mode_ds[6:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[:int(len(dataset) * percentage / 100)]]
        data_adv = dataset_selection
        data_cli = dataset
    elif mode_ds.startswith('split'):
        if mode_ds.startswith('splitn'):
            ndocs_adv = int(mode_ds[6:])
        else:
            percentage = 50 if mode_ds == 'split' else int(mode_ds[5:])
            assert 0 < percentage < 100
            ndocs_adv = int(len(dataset) * percentage / 100)
        permutation = np.random.permutation(len(dataset))
        data_adv = [dataset[i] for i in permutation[:ndocs_adv]]
        data_cli = [dataset[i] for i in permutation[ndocs_adv:]]
    else:
        raise ValueError("Dataset split mode '{:s}' not allowed".format(mode_ds))

    # Load query frequency info
    if freq_name == 'file':
        freq_adv, freq_cli, freq_real = build_frequencies_from_file(dataset_name, chosen_kw_indices, keywords, aux_dataset_info, mode_fs)
    elif freq_name.startswith('zipf'):
        shift = int(freq_name[5:]) if freq_name.startswith('zipfs') else 0  # zipfs200 is a zipf with 200 shift
        aux = np.array([1 / (i + shift + 1) for i in range(nkw)])
        freq_adv = freq_cli = freq_real = aux / np.sum(aux)
    elif freq_name == 'none':
        freq_adv, freq_cli, freq_real = None, None, np.ones(nkw) / nkw
    else:
        raise ValueError("Frequency name '{:s}' not implemented yet".format(freq_name))

    full_data_adv = {'dataset': data_adv,
                     'keywords': chosen_kw_indices,
                     'frequencies': freq_adv,
                     'mode_query': gen_params['mode_query']}
    full_data_client = {'dataset': data_cli,
                        'keywords': chosen_kw_indices,
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

    # full_data_adv, full_data_client, freq_real = generate_train_test_data(exp_param.gen_params)
    # v_print("Generated train-test data: adv dataset {:d}, client dataset {:d} ({:.1f} secs)".format(len(full_data_adv['dataset']),
    #                                                                                                 len(full_data_client['dataset']),
    #                                                                                                 time.time() - t0))

    # real_queries = generate_keyword_queries(exp_param.gen_params['mode_query'], freq_real, exp_param.gen_params['nqr'])
    # v_print("Generated {:d} real queries ({:.1f} secs)".format(len(real_queries), time.time() - t0))

    # observations, bw_overhead, real_and_dummy_queries = generate_observations(full_data_client, exp_param.def_params, real_queries)
    # v_print("Applied defense ({:.1f} secs)".format(time.time() - t0))
    
    full_data_adv, observations, real_and_dummy_queries = generate_page_observations(exp_param.gen_params)
    print("Generated observations ({:.1f} secs)".format(time.time() - t0))
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
