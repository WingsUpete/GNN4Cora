import os
import sys

import torch

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

import Config


class CDataSet(DGLDataset):
    """
    CDataSet loads from the Coda Dataset and process it before feeding into the trainer
    """
    def __init__(self, data_dir=Config.DATA_DIR_DEFAULT):
        super().__init__(name='cora')
        self.data_dir = data_dir

    def process(self):
        """
        Load and process raw data from disk
        """
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Provides the idx-th training sample
        :param idx: index of the training sample
        :return: the training sample which is a single citation graph
        """
        return None
