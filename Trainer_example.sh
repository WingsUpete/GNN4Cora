#!/bin/sh

### Preprocess ###
cd preprocess || exit
python CoraPreprocessor.py -dr ../data/cora/
cd ../



### Which Data to Use ###
datapath=data/cora
gid=0



### MLP ###
python Trainer.py -dr $datapath -gid $gid -m trainNeval -net MLP -tag MLP
#python Trainer.py -dr data/cora/ -m eval -e records/models/20220527_21_54_53.pth -tag MLP

### GCN ###
python Trainer.py -dr $datapath -gid $gid -m trainNeval -net GCN -tag GCN
#python Trainer.py -dr data/cora/ -m eval -e records/models/20220527_21_54_43.pth -tag GCN

### GAT ###
python Trainer.py -dr $datapath -gid $gid -m trainNeval -net GAT -tag GAT
#python Trainer.py -dr data/cora/ -m eval -e records/models/20220527_21_54_18.pth -tag GAT

### GaAN ###
python Trainer.py -dr $datapath -gid $gid -m trainNeval -net GaAN -tag GaAN
#python Trainer.py -dr data/cora/ -m eval -e records/models/20220527_21_36_39.pth -tag GaAN
