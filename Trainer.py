import os
import sys
import argparse
import time
import random

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

from CDataSet import CDataSet
from model import GCN, GAT, GaAN
from utils import Logger

import Config
if Config.CHECK_GRADS:
    torch.autograd.set_detect_anomaly(True)


def train(lr=Config.LEARNING_RATE_DEFAULT, bs=Config.BATCH_SIZE_DEFAULT, ep=Config.MAX_EPOCHS_DEFAULT,
          eval_freq=Config.EVAL_FREQ_DEFAULT, opt=Config.OPTIMIZER_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
          use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
          data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False),
          model=Config.NETWORK_DEFAULT,
          model_save_dir=Config.MODEL_SAVE_DIR_DEFAULT, train_type=Config.TRAIN_TYPE_DEFAULT,
          hidden_dim=Config.HIDDEN_DIM_DEFAULT, feat_dim=Config.FEAT_DIM_DEFAULT,
          retrain_model_path=Config.RETRAIN_MODEL_PATH_DEFAULT, loss_function=Config.LOSS_FUNC_DEFAULT):
    """
        Train and save the model
        1. Setup
        2. Train once on the training set
        3. Every once in a while, evaluate the model on the validation set and save the currently best version
    """
    # Create Device
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = CDataSet(data_dir)
    cgraph = dataset[0]
    # logr.log('> Training batches: {}, Validation batches: {}\n'.format(len(trainloader), len(validloader)))   # TODO: data info

    # Initialize the Model
    if train_type == 'retrain':
        logr.log('> Loading the Pretrained Model: {}, Train type = {}\n'.format(retrain_model_path, train_type))
        net = torch.load(retrain_model_path, map_location=device)
    else:
        logr.log('> Initializing the Training Model: {}, Train type = {}\n'.format(model, train_type))
        if model == 'GCN':
            net = GCN(in_dim=feat_dim, out_dim=hidden_dim,
                      use_pre_w=Config.USE_PRE_W_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT)
        elif model == 'GAT':
            net = GAT(in_dim=feat_dim, out_dim=hidden_dim,
                      use_pre_w=Config.USE_PRE_W_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
                       num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
        elif model == 'GaAN':
            net = GaAN(in_dim=feat_dim, out_dim=hidden_dim,
                       use_pre_w=Config.USE_PRE_W_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
                       num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
        else:
            # Default: GaAN
            net = GaAN(in_dim=feat_dim, out_dim=hidden_dim,
                       use_pre_w=Config.USE_PRE_W_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
                       num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)

    logr.log('> Model Structure:\n{}\n'.format(net))
    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Select Optimizer
    logr.log('> Constructing the Optimizer: {}\n'.format(opt))
    if opt == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)  # Adam + L2 Norm
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)  # Default: Adam + L2 Norm

    # Loss Function
    logr.log('> Using {} as the Loss Function.\n'.format(loss_function))
    if loss_function == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()   # Default: CrossEntropyLoss

    # Model Saving Directory
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    # Summarize Info
    logr.log('\nlearning_rate = {}, epochs = {}, num_workers = {}\n'.format(lr, ep, num_workers))
    logr.log('eval_freq = {}, batch_size = {}, optimizer = {}\n'.format(eval_freq, bs, opt))

    # Start Training
    logr.log('\nStart Training!\n')
    logr.log('------------------------------------------------------------------------\n')

    for epoch_i in range(ep):
        # Train one round
        net.train()
        # TODO: train and get result



def evaluate(model_name, bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
             use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
             data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False)):
    """
        Evaluate using saved best model (Note that this is a Test API)
        1. Re-evaluate on the validation set
        2. Re-evaluate on the test set
        The evaluation metrics include xxx
    """
    pass


if __name__ == '__main__':
    """ 
        Usage Example:
        python Trainer.py -dr data/cora/ -c 4 -m trainNeval -net GaAN
    """
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=Config.LEARNING_RATE_DEFAULT, help='Learning rate, default = {}'.format(Config.LEARNING_RATE_DEFAULT))
    parser.add_argument('-me', '--max_epochs', type=int, default=Config.MAX_EPOCHS_DEFAULT, help='Number of epochs to run the trainer, default = {}'.format(Config.MAX_EPOCHS_DEFAULT))
    parser.add_argument('-ef', '--eval_freq', type=int, default=Config.EVAL_FREQ_DEFAULT, help='Frequency of evaluation on the validation set, default = {}'.format(Config.EVAL_FREQ_DEFAULT))
    parser.add_argument('-bs', '--batch_size', type=int, default=Config.BATCH_SIZE_DEFAULT, help='Size of a minibatch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-lf', '--loss_function', type=str, default=Config.LOSS_FUNC_DEFAULT, help='Specify which loss function to use, default = {}'.format(Config.LOSS_FUNC_DEFAULT))
    parser.add_argument('-opt', '--optimizer', type=str, default=Config.OPTIMIZER_DEFAULT, help='Optimizer to be used [ADAM], default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-m', '--mode', type=str, default=Config.MODE_DEFAULT, help='Specify which mode the model runs in (train, eval, trainNeval), default = {}'.format(Config.MODE_DEFAULT))
    parser.add_argument('-tt', '--train_type', type=str, default=Config.TRAIN_TYPE_DEFAULT, help='Specify train mode [normal, pretrain, retrain], default = {}'.format(Config.TRAIN_TYPE_DEFAULT))
    # model structure
    parser.add_argument('-net', '--network', type=str, default=Config.NETWORK_DEFAULT, help='Specify which model/network to use, default = {}'.format(Config.NETWORK_DEFAULT))
    parser.add_argument('-fd', '--feature_dim', type=int, default=Config.FEAT_DIM_DEFAULT, help='Specify the feature dimension, default = {}'.format(Config.FEAT_DIM_DEFAULT))
    parser.add_argument('-hd', '--hidden_dim', type=int, default=Config.HIDDEN_DIM_DEFAULT, help='Specify the hidden dimension, default = {}'.format(Config.HIDDEN_DIM_DEFAULT))
    # data handling
    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-gid', '--gpu_id', type=int, default=Config.GPU_ID_DEFAULT, help='Specify which GPU to use, default = {}'.format(Config.GPU_ID_DEFAULT))
    # auxiliary locations
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))
    parser.add_argument('-tag', '--tag', type=str, default=Config.TAG_DEFAULT, help='Name tag for the model, default = {}'.format(Config.TAG_DEFAULT))
    parser.add_argument('-md', '--model_save_dir', type=str, default=Config.MODEL_SAVE_DIR_DEFAULT, help='Specify the location of network to be saved, default = {}'.format(Config.MODEL_SAVE_DIR_DEFAULT))
    # retraining & evaluation
    parser.add_argument('-r', '--retrain_model_path', type=str, default=Config.RETRAIN_MODEL_PATH_DEFAULT, help='Specify the location of the model to be retrained if train type is retrain, default = {}'.format(Config.RETRAIN_MODEL_PATH_DEFAULT))
    parser.add_argument('-e', '--eval', type=str, default=Config.EVAL_DEFAULT, help='Specify the location of saved network to be loaded for evaluation, default = {}'.format(Config.EVAL_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir, comment='%s_%s' % (FLAGS.tag, FLAGS.mode)) \
        if FLAGS.log_dir else Logger(activate=False)

    # Controls reproducibility
    if Config.RAND_SEED:
        random.seed(Config.RAND_SEED)
        torch.manual_seed(Config.RAND_SEED)
        logger.log('> Seed: %d\n' % Config.RAND_SEED)

    working_mode = FLAGS.mode
    if working_mode == 'train':
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
              data_dir=FLAGS.data_dir, logr=logger,
              model=FLAGS.network,
              model_save_dir=FLAGS.model_save_dir, train_type=FLAGS.train_type,
              hidden_dim=FLAGS.hidden_dim, feat_dim=FLAGS.feature_dim,
              retrain_model_path=FLAGS.retrain_model_path, loss_function=FLAGS.loss_function)
        logger.close()
    elif working_mode == 'eval':
        eval_file = FLAGS.eval
        # Abnormal: file not found
        if (not eval_file) or (not os.path.isfile(eval_file)):
            sys.stderr.write('File for evaluation not found, please check!\n')
            logger.close()
            exit(-1)
        # Normal
        evaluate(eval_file, bs=FLAGS.batch_size, num_workers=FLAGS.cores,
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
                 data_dir=FLAGS.data_dir, logr=logger)
        logger.close()
    elif working_mode == 'trainNeval':
        # First train then eval
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
              data_dir=FLAGS.data_dir, logr=logger,
              model=FLAGS.network,
              model_save_dir=FLAGS.model_save_dir, train_type=FLAGS.train_type,
              hidden_dim=FLAGS.hidden_dim, feat_dim=FLAGS.feature_dim,
              retrain_model_path=FLAGS.retrain_model_path, loss_function=FLAGS.loss_function)

        saved_model_path = os.path.join(Config.MODEL_SAVE_DIR_DEFAULT, '%s.pth' % logger.time_tag)
        logger.log('\n')

        evaluate(saved_model_path, bs=FLAGS.batch_size, num_workers=FLAGS.cores,
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
                 data_dir=FLAGS.data_dir, logr=logger)

        logger.close()
    else:
        sys.stderr.write('Please specify the running mode (train/eval/trainNeval)\n')
        logger.close()
        exit(-2)
