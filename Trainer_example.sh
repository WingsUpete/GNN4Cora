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

### GCN ###
python Trainer.py -dr $datapath -gid $gid -m trainNeval -net GCN -tag GCN

### GAT ###
python Trainer.py -dr $datapath -gid $gid -m trainNeval -net GAT -tag GAT

### GaAN ###
python Trainer.py -dr $datapath -gid $gid -m trainNeval -net GaAN -tag GaAN
