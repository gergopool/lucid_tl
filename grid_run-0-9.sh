#!/bin/bash
for i in {0..9}
do
    CUDA_VISIBLE_DEVICES=$((i/10)) python src/bin/train.py config/train/cka_alter/in$i-gn$i.ini > $i.out 
done