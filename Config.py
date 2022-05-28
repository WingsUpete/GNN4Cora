# DEBUG FLAGS
TRAIN_JUST_ONE_ROUND = False
PROFILE = False
CHECK_GRADS = False
# RAND_SEED = None
RAND_SEED = 6666666

# Basic
FEAT_DIM_DEFAULT = 1433
HIDDEN_DIM_DEFAULT = 128
NUM_CLASSES = 7

NUM_HEADS_DEFAULT = 3

LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 75
EVAL_FREQ_DEFAULT = 5
MAX_NORM_DEFAULT = 10.0

USE_GPU_DEFAULT = 1
GPU_ID_DEFAULT = 0
DATA_SPLIT_MODES = ['imbalance', 'balance']
DATA_SPLIT_MODE_DEFAULT = 'imbalance'  # balance, imbalance
TRAIN_VALID_TEST_SPLIT_NUM_DEFAULT = [140, 500, 1000]   # quantity, for balance
TRAIN_VALID_TEST_SPLIT_RATIO_DEFAULT = [0.1, 0.3, 0.6]  # ratio, for imbalance

OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_DECAY_DEFAULT = 0.01

LOSS_FUNC_DEFAULT = 'CrossEntropyLoss'

DATA_DIR_DEFAULT = 'data/cora/'
LOG_DIR_DEFAULT = 'log/'
TAG_DEFAULT = None
MODEL_SAVE_DIR_DEFAULT = 'model_save/'
EVAL_DEFAULT = 'eval.pth'   # should be a model file name

NETWORK_DEFAULT = 'GaAN'
NETWORKS = ['GCN', 'GAT', 'GaAN']
MODE_DEFAULT = 'trainNeval'
BLK_SIZE_DEFAULT = 2
MERGE_HEAD_MODE_DEFAULT = 'cat'

VIEWS = ['citing', 'cited', 'both', 'double']
VIEW_DEFAULT = 'both'   # citing, cited, both, double
NUM_VIEW_DEFAULT = 2 if VIEW_DEFAULT == 'double' else 1

# Customize: DIY
# OUR_MODEL = 'OUR_MODEL'

MODELS_TO_EXAMINE = [
    ['MLP'],                    # baseline
    ['GCN', 'GAT', 'GaAN'],     # others
    # [OUR_MODEL],              # ours
    ['?',                       # variants
     ]
]
