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
sys.stderr.close()
sys.stderr = stderr

from CDataSet import CDataSet
from model import GCN, GAT, GaAN, CMLP
from utils import Logger, plot_grad_flow

import Config
if Config.CHECK_GRADS:
    torch.autograd.set_detect_anomaly(True)


def train(lr=Config.LEARNING_RATE_DEFAULT, ep=Config.MAX_EPOCHS_DEFAULT,
          eval_freq=Config.EVAL_FREQ_DEFAULT, opt=Config.OPTIMIZER_DEFAULT,
          use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
          data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False),
          model=Config.NETWORK_DEFAULT,
          model_save_dir=Config.MODEL_SAVE_DIR_DEFAULT,
          feat_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES,
          loss_function=Config.LOSS_FUNC_DEFAULT):
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
    cgraphs = dataset[0]
    if device:
        for i in range(len(cgraphs)):
            cgraphs[i] = cgraphs[i].to(device)
        if dataset.need_loss_weights:
            dataset.loss_weights = dataset.loss_weights.to(device)
        logr.log('> Data sent to {}\n'.format(device))
    logr.log('> view: %s\n' % dataset.view)
    logr.log('> num_nodes: %d, num_edges: %s\n' % (dataset.meta['num_nodes'], str(dataset.meta['num_edges'])))
    logr.log('> num_feats: %d, num_classes: %d\n' % (dataset.meta['num_feats'], dataset.meta['num_classes']))
    logr.log('> num_samples: training = %d, validation = %d, test = %d\n' % (dataset.num_train, dataset.num_valid, dataset.num_test))
    logr.log('> train_set_imbalance: %s\n' % dataset.train_imbalance_record)
    if dataset.need_loss_weights:
        logr.log('> loss_weights: %s\n' % dataset.loss_weights)

    # Initialize the Model
    logr.log('> Initializing the Training Model: {}\n'.format(model))
    if model == 'GCN':
        net = GCN(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                  num_view=Config.NUM_VIEW_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT)
    elif model == 'GAT':
        net = GAT(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                  num_view=Config.NUM_VIEW_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
                  num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
    elif model == 'GaAN':
        net = GaAN(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                   num_view=Config.NUM_VIEW_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
                   num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
    elif model == 'MLP':
        net = CMLP(in_dim=feat_dim, hidden_dim_ref=hidden_dim, out_dim=out_dim)
    else:
        # Default: GaAN
        net = GaAN(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                   num_view=Config.NUM_VIEW_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
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
    if dataset.need_loss_weights:
        logr.log('> Using loss weights for cost rescaling.\n')
        loss_weights = dataset.loss_weights
    else:
        loss_weights = None
    if loss_function == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weights)   # Default: CrossEntropyLoss

    # Model Saving Directory
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    # Summarize Info
    logr.log('\nlearning_rate = {}, epochs = {}\n'.format(lr, ep))
    logr.log('eval_freq = {}, optimizer = {}\n'.format(eval_freq, opt))

    # Start Training
    logr.log('\nStart Training!\n')
    logr.log('------------------------------------------------------------------------\n')

    max_eval_acc = 0.0
    features = cgraphs[-1].ndata['feat']
    labels = cgraphs[-1].ndata['label']
    train_mask = cgraphs[-1].ndata['train_mask']
    valid_mask = cgraphs[-1].ndata['valid_mask']
    test_mask = cgraphs[-1].ndata['test_mask']
    for epoch_i in range(ep):
        # Train one round
        net.train()
        time_start_train = time.time()
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()

        # Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

        optimizer.zero_grad()

        if Config.PROFILE:
            with profiler.profile(profile_memory=True, use_cuda=True) as prof:
                with profiler.record_function('model_inference'):
                    logits = net(cgraphs, features)
                    pred = logits.argmax(dim=1)
            logr.log(prof.key_averages().table(sort_by="cuda_time_total"))
            exit(100)

        logits = net(cgraphs, features)
        pred = logits.argmax(dim=1)

        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()

        if Config.CHECK_GRADS:
            plot_grad_flow(net.named_parameters())

        optimizer.step()

        time_end_train = time.time()
        total_train_time = (time_end_train - time_start_train)
        with torch.no_grad():
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            logr.log('Training Round %d: loss = %.6f, time_cost = %.4f sec, acc = %.4f%%\n' %
                     (epoch_i + 1, loss.item(), total_train_time, train_acc * 100))

        # eval_freq: Evaluate on validation set
        if (epoch_i + 1) % eval_freq == 0:
            net.eval()
            with torch.no_grad():
                valid_acc = (pred[valid_mask] == labels[valid_mask]).float().mean()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
                logr.log('!!! Evaluation: valid_acc = %.4f%%, test_acc = %.4f%%\n' % (valid_acc * 100, test_acc * 100))
                # Select current best model
                if epoch_i >= 10 and valid_acc > max_eval_acc:
                    max_eval_acc = valid_acc
                    model_name = os.path.join(model_save_dir, '{}.pth'.format(logr.time_tag))
                    torch.save(net, model_name)
                    logr.log('Model: {} has been saved since it achieves higher validation accuracy.\n'.format(model_name))

        if Config.TRAIN_JUST_ONE_ROUND:
            if epoch_i == 0:    # DEBUG
                break

    # End Training
    logr.log('> Training finished.\n')


def evaluate(model_name,
             use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
             data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False)):
    """
        Evaluate using saved best model (Note that this is a Test API)
        1. Re-evaluate on the training set
        2. Re-evaluate on the validation set
        3. Evaluate on the test set
        The evaluation metrics include "accuracy"
    """
    # Create Device
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = CDataSet(data_dir)
    cgraphs = dataset[0]
    if device:
        for i in range(len(cgraphs)):
            cgraphs[i] = cgraphs[i].to(device)
        logr.log('> Data sent to {}\n'.format(device))
    logr.log('> view: %s\n' % dataset.view)
    logr.log('> num_nodes: %d, num_edges: %s\n' % (dataset.meta['num_nodes'], str(dataset.meta['num_edges'])))
    logr.log('> num_feats: %d, num_classes: %d\n' % (dataset.meta['num_feats'], dataset.meta['num_classes']))
    logr.log('> num_samples: training = %d, validation = %d, test = %d\n' % (dataset.num_train, dataset.num_valid, dataset.num_test))
    logr.log('> train_set_imbalance: %s\n' % dataset.train_imbalance_record)

    # Load Model
    logr.log('> Loading {}\n'.format(model_name))
    net = torch.load(model_name, map_location=device)
    logr.log('> Model Structure:\n{}\n'.format(net))
    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Evaluate now
    features = cgraphs[-1].ndata['feat']
    labels = cgraphs[-1].ndata['label']
    valid_mask = cgraphs[-1].ndata['valid_mask']
    test_mask = cgraphs[-1].ndata['test_mask']
    net.eval()
    logits = net(cgraphs, features)
    pred = logits.argmax(dim=1)
    valid_acc = (pred[valid_mask] == labels[valid_mask]).float().mean()
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    logr.log('> Evaluation Results: valid_acc = %.4f%%, test_acc = %.4f%%\n' % (valid_acc * 100, test_acc * 100))

    # End Evaluation
    logr.log('> Evaluation finished.\n')


if __name__ == '__main__':
    """ 
        Usage Example:
        python Trainer.py -dr data/cora/ -m trainNeval -net GaAN -tag GaAN
    """
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=Config.LEARNING_RATE_DEFAULT, help='Learning rate, default = {}'.format(Config.LEARNING_RATE_DEFAULT))
    parser.add_argument('-me', '--max_epochs', type=int, default=Config.MAX_EPOCHS_DEFAULT, help='Number of epochs to run the trainer, default = {}'.format(Config.MAX_EPOCHS_DEFAULT))
    parser.add_argument('-ef', '--eval_freq', type=int, default=Config.EVAL_FREQ_DEFAULT, help='Frequency of evaluation on the validation set, default = {}'.format(Config.EVAL_FREQ_DEFAULT))
    parser.add_argument('-lf', '--loss_function', type=str, default=Config.LOSS_FUNC_DEFAULT, help='Specify which loss function to use, default = {}'.format(Config.LOSS_FUNC_DEFAULT))
    parser.add_argument('-opt', '--optimizer', type=str, default=Config.OPTIMIZER_DEFAULT, help='Optimizer to be used [ADAM], default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-m', '--mode', type=str, default=Config.MODE_DEFAULT, help='Specify which mode the model runs in (train, eval, trainNeval), default = {}'.format(Config.MODE_DEFAULT))
    # model structure
    parser.add_argument('-net', '--network', type=str, default=Config.NETWORK_DEFAULT, help='Specify which model/network to use, default = {}'.format(Config.NETWORK_DEFAULT))
    parser.add_argument('-fd', '--feature_dim', type=int, default=Config.FEAT_DIM_DEFAULT, help='Specify the feature dimension, default = {}'.format(Config.FEAT_DIM_DEFAULT))
    parser.add_argument('-hd', '--hidden_dim', type=int, default=Config.HIDDEN_DIM_DEFAULT, help='Specify the hidden dimension, default = {}'.format(Config.HIDDEN_DIM_DEFAULT))
    parser.add_argument('-od', '--out_dim', type=int, default=Config.NUM_CLASSES, help='Specify the output dimension, default = {}'.format(Config.NUM_CLASSES))
    # data handling
    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-gid', '--gpu_id', type=int, default=Config.GPU_ID_DEFAULT, help='Specify which GPU to use, default = {}'.format(Config.GPU_ID_DEFAULT))
    # auxiliary locations
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))
    parser.add_argument('-tag', '--tag', type=str, default=Config.TAG_DEFAULT, help='Name tag for the model, default = {}'.format(Config.TAG_DEFAULT))
    parser.add_argument('-md', '--model_save_dir', type=str, default=Config.MODEL_SAVE_DIR_DEFAULT, help='Specify the location of network to be saved, default = {}'.format(Config.MODEL_SAVE_DIR_DEFAULT))
    # evaluation
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
        train(lr=FLAGS.learning_rate, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
              data_dir=FLAGS.data_dir, logr=logger,
              model=FLAGS.network,
              model_save_dir=FLAGS.model_save_dir,
              feat_dim=FLAGS.feature_dim, hidden_dim=FLAGS.hidden_dim, out_dim=FLAGS.out_dim,
              loss_function=FLAGS.loss_function)
        logger.close()
    elif working_mode == 'eval':
        eval_file = FLAGS.eval
        # Abnormal: file not found
        if (not eval_file) or (not os.path.isfile(eval_file)):
            sys.stderr.write('File for evaluation not found, please check!\n')
            logger.close()
            exit(-1)
        # Normal
        evaluate(eval_file,
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
                 data_dir=FLAGS.data_dir, logr=logger)
        logger.close()
    elif working_mode == 'trainNeval':
        # First train then eval
        train(lr=FLAGS.learning_rate, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
              data_dir=FLAGS.data_dir, logr=logger,
              model=FLAGS.network,
              model_save_dir=FLAGS.model_save_dir,
              feat_dim=FLAGS.feature_dim, hidden_dim=FLAGS.hidden_dim, out_dim=FLAGS.out_dim,
              loss_function=FLAGS.loss_function)

        saved_model_path = os.path.join(Config.MODEL_SAVE_DIR_DEFAULT, '%s.pth' % logger.time_tag)
        logger.log('\n')

        evaluate(saved_model_path,
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
                 data_dir=FLAGS.data_dir, logr=logger)

        logger.close()
    else:
        sys.stderr.write('Please specify the running mode (train/eval/trainNeval)\n')
        logger.close()
        exit(-2)
