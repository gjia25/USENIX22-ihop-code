import numpy as np
import utils
from collections import defaultdict


def generate_observations(full_data_client, def_params, real_queries):
    """kw_id is the id wrt to this run (e.g., 0, 1, 2, ...nkw)
    keywords are ids wrt the original full dataset (i.e., they represent the actual keyword)
    We need gen_params to get ground truth queries (if any)
    """

    observations = {}
    traces = []
    dataset = full_data_client['dataset']
    keywords = np.array(full_data_client['keywords'])
    nkw = len(keywords)
    correct_mapping = None

    # Two ways of doing this, the second one is way faster!
    # First:
    # inverted_index = {kw_id: [doc_id for doc_id, doc_kws in enumerate(dataset) if keywords[kw_id] in doc_kws] for kw_id in range(nkw)}
    # Second:
    inverted_index = defaultdict(list)
    kw_to_kw_id = {kw: kw_id for kw_id, kw in enumerate(keywords)}
    for doc_id, doc_kws in enumerate(dataset):
        for kw in set(doc_kws) & set(keywords):
            inverted_index[kw_to_kw_id[kw]].append(doc_id)

    if def_params['name'] == 'none':

        trace_type = 'ap_unique'
        token_ids = np.random.permutation(np.max(real_queries) + 1)
        for kw_id in real_queries:
            traces.append((token_ids[kw_id], inverted_index[kw_id]))
        bw_overhead = 1
        real_and_dummy_queries = real_queries
        correct_mapping = {token_id: kw_id for kw_id, token_id in enumerate(token_ids)}

    elif def_params['name'] == 'clrz':

        trace_type = 'ap_unique'
        token_ids = np.random.permutation(np.max(real_queries) + 1)
        tpr, fpr = def_params['tpr'], def_params['fpr']
        obf_inverted_index = {}
        for kw_id in range(nkw):
            coin_flips = np.random.rand(len(dataset))
            obf_inverted_index[kw_id] = [doc_id for doc_id, doc_kws in enumerate(dataset) if
                                         (keywords[kw_id] in doc_kws and coin_flips[doc_id] < tpr) or
                                         (keywords[kw_id] not in doc_kws and coin_flips[doc_id] < fpr)]

        ndocs_retrieved = 0
        ndocs_real = 0
        for kw_id in real_queries:
            traces.append((token_ids[kw_id], obf_inverted_index[kw_id]))
            ndocs_retrieved += len(obf_inverted_index[kw_id])
            ndocs_real += len(inverted_index[kw_id])
        bw_overhead = ndocs_retrieved / ndocs_real
        real_and_dummy_queries = real_queries

    elif def_params['name'] == 'osse':

        trace_type = 'ap_osse'
        tpr, fpr = def_params['tpr'], def_params['fpr']
        ndocs_retrieved = 0
        ndocs_real = 0
        for kw_id in real_queries:
            coin_flips = np.random.rand(len(dataset))
            traces.append([doc_id for doc_id, doc_kws in enumerate(dataset) if
                           (keywords[kw_id] in doc_kws and coin_flips[doc_id] < tpr) or
                           (keywords[kw_id] not in doc_kws and coin_flips[doc_id] < fpr)])
            ndocs_retrieved += len(traces[-1])
            ndocs_real += len(inverted_index[kw_id])
        bw_overhead = ndocs_retrieved / ndocs_real
        observations['n_distinct'] = len(set(real_queries))
        real_and_dummy_queries = real_queries

    elif def_params['name'] == 'pancake' or def_params['name'] == 'swat':

        trace_type = 'tok_vol'
        freal = utils.get_steady_state(full_data_client['frequencies']) if full_data_client['frequencies'].ndim == 2 else full_data_client['frequencies']
        prob_reals, prob_dummies, replicas_per_kw = utils.compute_pancake_parameters(nkw, freal)

        # some imprecision in calculating prob_reals above causes some values to be very small negative numbers,
        # which then was causing errors later in the attack when we expect probabilities to be nonnegative
        prob_reals[prob_reals<0] = -prob_reals[prob_reals<0]

        permutation = np.random.permutation(2 * nkw)
        aux = [0] + list(np.cumsum(replicas_per_kw, dtype=int))
        kw_id_to_replica = [tuple(permutation[aux[i]: aux[i + 1]]) for i in range(len(aux) - 1)] # kw_id_to_replica[id] is a tuple w all replicas matching kw id

        bw_overhead = 3

        if def_params['name'] == 'swat': # add SWAT sampling pool
            traces, real_and_dummy_queries = generate_theta_decorr_obs(nkw, kw_id_to_replica, real_queries, prob_reals, prob_dummies)
        else: # just pancake
            nq = len(real_queries)
            perm = np.random.permutation(3 * nq)
            separation = np.random.binomial(3 * nq, 0.5)
            indices_real_slots, indices_dummy_slots = perm[:separation], perm[separation:]
            indices_real_slots.sort()
            indices_for_each_true_message = []
            real_slots_copy = indices_real_slots.copy()
            for i in range(0, 3 * nq, 3):   # maximum number of real slots chosen before reaching query i in nq is i/3 slots;
                                            # models injecting a new query from real_queries once every 3 requests
                try:
                    index = next(filter(lambda x: real_slots_copy[x] >= i, range(len(real_slots_copy))))
                except StopIteration:
                    break
                indices_for_each_true_message.append(real_slots_copy[index])
                real_slots_copy = real_slots_copy[(index + 1):]

            trace_no_replicas = -np.ones(3 * nq, dtype=int)
            trace_no_replicas[indices_dummy_slots] = np.random.choice(nkw + 1, len(indices_dummy_slots), replace=True, p=prob_dummies)
            trace_no_replicas[indices_real_slots] = np.random.choice(nkw + 1, len(indices_real_slots), replace=True, p=prob_reals)
            trace_no_replicas[indices_for_each_true_message] = real_queries[:len(indices_for_each_true_message)]

            for kw_id in trace_no_replicas:
                traces.append((np.random.choice(kw_id_to_replica[kw_id]), 1))  # Volume is 1

            real_and_dummy_queries = trace_no_replicas
        
        correct_mapping = {int(replica): kw_id for kw_id, replicas in enumerate(kw_id_to_replica) for replica in replicas}

    elif def_params['name'] == 'waffle':
        raise NotImplementedError("Waffle defense not yet implemented")

    else:
        raise ValueError("Defense {:s} not implemented".format(def_params['name']))

    observations['traces'] = traces
    observations['trace_type'] = trace_type
    observations['ndocs'] = len(dataset)

    return observations, bw_overhead, real_and_dummy_queries, correct_mapping


def generate_theta_decorr_obs(nkw, kw_id_to_replica, real_queries, prob_reals, prob_dummies):
    # SWAT query pool implementation based on original codebase: https://github.com/CongGroup/SWAT/blob/main/query_decorrelation.py
    from config import NQR, THETA, SAMPLING_FUNC

    que = SamplingPool(THETA, SAMPLING_FUNC)
    q_szs = {} # for debug
    cnt = 0 # counter; essentially a timestamp

    real_and_dummy_queries = []
    detailedTranscript = []

    traces = []

    print("nkw:", nkw)
    print("real_queries[:25]", real_queries[:25])

    for q in real_queries:
        if (len(real_and_dummy_queries) >= NQR): continue

        # for debug, delete
        if (len(real_and_dummy_queries)%100_000 == 0): print(len(real_and_dummy_queries), "observations generated")

        que.put((q, cnt))
        latencies = []
        for i in range(3):
            delta = 0.5
            type = np.random.choice([0, 1], p=[delta, 1-delta])
            if type == 0:
                idx = np.random.choice(nkw+1, p=prob_dummies)
            else:
                if que.qsize() > THETA:
                    idx, cnt_ = que.get()
                    latencies.append(cnt - cnt_)
                    type = 0 # not a real query from transcript
                else:
                    idx = np.random.choice(nkw+1, p=prob_reals)

            key_ = np.random.choice(kw_id_to_replica[idx])

            real_and_dummy_queries.append(idx)
            traces.append((key_, 1))

            # for debug
            detailedTranscript.append( (idx, key_, type == True) )
            if que.qsize() not in q_szs: q_szs[que.qsize()] = 0
            q_szs[que.qsize()] += 1

        cnt+=1

    print("First 100 queries (key, replica, isRealQueryFromTranscript):", detailedTranscript[:100])

    return traces, np.array(real_and_dummy_queries)

class SamplingPool:
    def __init__(self, initNum = 0, update = None):
        self.pool = []
        self.initNum = initNum
        self.update = update
        self.weight = []
        self.q_szs = {}

    def get(self):
        ret = 0
        if self.update is None:
            t = np.random.randint(0, len(self.pool))
            ret = self.pool.pop(t)
        elif self.update == "Linear":
            t = np.random.choice(np.arange(len(self.pool)),
                                p=np.array(self.weight) / np.sum(self.weight))
            ret = self.pool.pop(t)
            self.weight.pop(t)
            self.weight = [x + 3 for x in self.weight]
        elif self.update == "Exp":
            t = np.random.choice(np.arange(len(self.pool)),
                                p=np.array(self.weight) / np.sum(self.weight))
            ret = self.pool.pop(t)
            self.weight.pop(t)
            self.weight = [x * 5 for x in self.weight]
        return ret

    def qsize(self):
        return len(self.pool)

    def empty(self) -> bool:
        return len(self.pool) <= self.initNum

    def put(self, x):
        self.pool.append(x)
        self.weight.append(1)
