RAW_DATASET_FOLDER = 'datasets_raw'
PRE_DATASET_FOLDER = 'datasets_pre'
PRO_DATASET_FOLDER = '../datasets_pro'

# debug.py
NKW = 250
NQR = 1_000_000
NITERS = 10_000
NITER_LIST = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
PFREE = 0.25

NUM_TARGETS_PAIRS = 50  # number of target kws for pairs attack
THETA = 1               # can't sample from pool until pool size > THETA (following initial SWAT implementation in decorr.py)
SAMPLING_FUNC = "Exp"   # sampling pool strategy (None, "Linear", "Exp")

NRUNS = 5

# experiment.py
CORR_LEVEL = 'mid'  # 'high': each kw only transitions to one (first/random) doc containing it - see HIGH_CORR_PERMUTE
                    # 'mid' : each kw can transition to any doc containing it, but weighted exponentially
                    # 'low' : each kw can transition to any doc containing it (weighted equally)
HIGH_CORR_PERMUTE = True # If CORR_LEVEL = 'high' and HIGH_CORR_PERMUTE = True, each kw only transitions to random doc containing it
                          # If False, each kw only transitions to first doc containing it

BASE_SEED = 58 # debug.py runs deterministically; introduce randomness in case a run gets interrupted and we want to restart without redoing all of the previously-done randomness
