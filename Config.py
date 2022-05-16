# DEBUG FLAGS
TRAIN_JUST_ONE_BATCH = False
TRAIN_JUST_ONE_ROUND = False
PROFILE = False
CHECK_GRADS = False
# RAND_SEED = None
RAND_SEED = 66666

# Basic
FEAT_DIM_DEFAULT = 1443
HIDDEN_DIM_DEFAULT = 16
NUM_CLASSES = 7

NUM_HEADS_DEFAULT = 3

LEARNING_RATE_DEFAULT = 1e-2    # 0.01
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 5
MAX_NORM_DEFAULT = 10.0

BATCH_SIZE_DEFAULT = 16
WORKERS_DEFAULT = 4
USE_GPU_DEFAULT = 1
GPU_ID_DEFAULT = 0

OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_DECAY_DEFAULT = 0.01

LOSS_FUNC_DEFAULT = 'CrossEntropyLoss'
SMOOTH_L1_LOSS_BETA_DEFAULT = 10

DATA_DIR_DEFAULT = 'data/cora/'
LOG_DIR_DEFAULT = 'log/'
TAG_DEFAULT = None
MODEL_SAVE_DIR_DEFAULT = 'model_save/'
RETRAIN_MODEL_PATH_DEFAULT = 'pretrained_model.pth'
EVAL_DEFAULT = 'eval.pth'   # should be a model file name

NETWORK_DEFAULT = 'GaAN'
NETWORKS = ['GCN', 'GAT', 'GaAN']
MODE_DEFAULT = 'trainNeval'
TRAIN_TYPES = ['normal', 'pretrain', 'retrain']
TRAIN_TYPE_DEFAULT = 'normal'
USE_PRE_W_DEFAULT = False
BLK_SIZE_DEFAULT = 1
MERGE_HEAD_MODE_DEFAULT = 'cat'

# Customize: DIY
# OUR_MODEL = 'OUR_MODEL'

MODELS_TO_EXAMINE = [
    ['?'],                      # baseline
    ['GCN', 'GAT', 'GaAN'],     # others
    # [OUR_MODEL],              # ours
    ['?',                       # variants
     ]
]
