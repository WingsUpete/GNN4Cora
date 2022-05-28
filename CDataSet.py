import os
import sys
import json
import random

import torch

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
from dgl.data import DGLDataset
sys.stderr.close()
sys.stderr = stderr

import Config


class CDataSet(DGLDataset):
    """
    CDataSet loads from the Coda Dataset and process it before feeding into the trainer
    """
    def __init__(self, data_dir=Config.DATA_DIR_DEFAULT,
                 train_valid_test_ratio=Config.TRAIN_VALID_TEST_SPLIT_RATIO_DEFAULT,
                 view=Config.VIEW_DEFAULT):
        super().__init__(name='cora')
        self.split_mode = Config.DATA_SPLIT_MODE_DEFAULT
        if self.split_mode not in Config.DATA_SPLIT_MODES:
            sys.stderr.write('> [CDataSet:init] Unrecognized Train-Validation-Test split mode "%s".\n' % Config.DATA_SPLIT_MODE_DEFAULT)
            exit(-400)
        if self.split_mode == 'imbalance':  # casually group first split section
            if (train_valid_test_ratio[0] < 0 or train_valid_test_ratio[1] < 0 or train_valid_test_ratio[2] < 0) or \
                    (sum(train_valid_test_ratio) != 1.0):
                sys.stderr.write('> [CDataSet:init] Train-Validation-Test ratios (%.1f, %.1f, %.1f) are not valid. Use (%.1f, %.1f, %.1f) by default.\n' %
                                 (train_valid_test_ratio[0], train_valid_test_ratio[1], train_valid_test_ratio[2], Config.TRAIN_VALID_TEST_SPLIT_RATIO_DEFAULT[0], Config.TRAIN_VALID_TEST_SPLIT_RATIO_DEFAULT[1], Config.TRAIN_VALID_TEST_SPLIT_RATIO_DEFAULT[2]))
        elif self.split_mode == 'balance':  # force selecting 20 for each class, which is a classic approach
            pass
        else:
            exit(-401)

        self.data_dir = data_dir
        self.meta = json.load(open(os.path.join(self.data_dir, 'meta.json')))
        graphs, _ = dgl.load_graphs(os.path.join(self.data_dir, 'cora.dgl'))
        self.view = view
        self.graphs = self.select_graphs(graphs)

        if self.split_mode == 'imbalance':
            self.train_valid_test_split_ratio = train_valid_test_ratio
            self.num_train = int(self.meta['num_nodes'] * train_valid_test_ratio[0])
            self.num_valid = int(self.meta['num_nodes'] * train_valid_test_ratio[1])
            self.num_test = self.meta['num_nodes'] - self.num_train - self.num_valid
        elif self.split_mode == 'balance':
            self.num_train, self.num_valid, self.num_test = Config.TRAIN_VALID_TEST_SPLIT_NUM_DEFAULT
        else:
            exit(-402)

        self.train_imbalance_record = self.train_valid_test_split()
        # Calculate weights for loss rescaling
        self.need_loss_weights, self.loss_weights = self.cal_loss_weights()

    def select_graphs(self, graphs):
        if self.view not in Config.VIEWS:
            sys.stderr.write('> [CDataSet:select_graphs] Unrecognized view "%s", use "%s" instead.\n' % (self.view, Config.VIEW_DEFAULT))
            self.view = Config.VIEW_DEFAULT
        if self.view == 'double':
            self.meta['num_edges'] = self.meta['num_edges'][:2]
            return graphs[:2]
        else:
            self.meta['num_edges'] = [self.meta['num_edges'][self.meta['relationship_map'][self.view]]]
            return [graphs[self.meta['relationship_map'][self.view]]]

    def train_valid_test_split(self):
        train_mask = [False for _ in range(self.meta['num_nodes'])]
        valid_mask = [False for _ in range(self.meta['num_nodes'])]
        test_mask = [False for _ in range(self.meta['num_nodes'])]
        train_imbalance_record = {}
        for i in range(self.meta['num_classes']):
            train_imbalance_record[i] = 0

        random.seed(Config.RAND_SEED)
        temp_list = [i for i in range(self.meta['num_nodes'])]
        random.shuffle(temp_list)

        if self.split_mode == 'imbalance':
            for i in range(self.num_train):
                train_mask[temp_list[i]] = True
                # Summarize imbalance of the training set
                cur_label = self.graphs[-1].ndata['label'][temp_list[i]].item()
                train_imbalance_record[cur_label] += 1
            for i in range(self.num_valid):
                valid_mask[temp_list[self.num_train + i]] = True
            for i in range(self.num_test):
                test_mask[temp_list[self.num_train + self.num_valid + i]] = True
        elif self.split_mode == 'balance':
            num_for_each_class = int(self.num_train / self.meta['num_classes'])
            train_cnt, valid_cnt, test_cnt = 0, 0, 0
            for node_id in temp_list:
                if train_cnt == self.num_train and valid_cnt == self.num_valid and test_cnt == self.num_test:
                    break
                is_train = False
                if train_cnt < self.num_train:
                    cur_label = self.graphs[-1].ndata['label'][node_id].item()
                    if train_imbalance_record[cur_label] < num_for_each_class:  # selectable
                        train_mask[node_id] = True
                        train_imbalance_record[cur_label] += 1
                        train_cnt += 1
                        is_train = True
                if not is_train:
                    if valid_cnt < self.num_valid:
                        valid_mask[node_id] = True
                        valid_cnt += 1
                    elif test_cnt < self.num_test:
                        test_mask[node_id] = True
                        test_cnt += 1
        else:
            exit(-403)

        for i in range(len(self.graphs)):
            self.graphs[i].ndata['train_mask'] = torch.Tensor(train_mask).bool()
            self.graphs[i].ndata['valid_mask'] = torch.Tensor(valid_mask).bool()
            self.graphs[i].ndata['test_mask'] = torch.Tensor(test_mask).bool()

        return train_imbalance_record

    def cal_loss_weights(self):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        w = num_train / (num_classes * train_imbalance_records)
        :return:
        """
        loss_weights = torch.Tensor([self.train_imbalance_record[i] for i in range(self.meta['num_classes'])]).float()
        if torch.max(loss_weights) / torch.min(loss_weights) < 10.0:    # Not so imbalanced
            return False, None
        loss_weights = self.num_train / (self.meta['num_classes'] * loss_weights)
        return True, loss_weights

    def process(self):
        """
        Load and process raw data from disk. We have preprocessing stage, so no need to do anything here.
        """
        pass

    def __len__(self):
        """ There is only one graph for the Cora Dataset """
        return 1

    def __getitem__(self, idx):
        """
        Provides the idx-th training sample
        :param idx: index of the training sample
        :return: the training sample which is a single citation graph
        """
        return self.graphs

    def __str__(self):
        info_msg = 'view: %s\n' % self.view
        info_msg += 'num_nodes: %d\nnum_edges: %s\nnum_feats: %d\nnum_classes: %d\n' % \
                   (self.meta['num_nodes'], str(self.meta['num_edges']),
                    self.meta['num_feats'], self.meta['num_classes'])
        info_msg += 'num_training_samples: %d\nnum_validation_samples: %d\nnum_test_samples: %d\n' % \
                    (self.num_train, self.num_valid, self.num_test)
        info_msg += 'train_set_imbalance: %s\n' % self.train_imbalance_record
        if self.need_loss_weights:
            info_msg += 'loss_weights: %s\n' % self.loss_weights
        return info_msg


if __name__ == '__main__':
    ds = CDataSet(data_dir=Config.DATA_DIR_DEFAULT, view='double')
    print(ds)
