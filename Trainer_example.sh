#!/bin/sh

### Preprocess ###
cd preprocess || exit
python CoraPreprocessor.py -dr ../data/cora/
cd ../



### Which Data to Use ###
datapath=data/cora



### GCN ###
python Trainer.py -dr $datapath -c 4 -gid 0 -m trainNeval -net GCN -tag GCN

### GAT ###
python Trainer.py -dr $datapath -c 4 -gid 0 -m trainNeval -net GAT -tag GAT

### GaAN ###
python Trainer.py -dr $datapath -c 4 -gid 0 -m trainNeval -net GaAN -tag GaAN
