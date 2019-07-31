#!/bin/bash

#./scripts/train-resnet-32-parallel.sh 

for i in {15..33}
do
  ./scripts/train-resnet-32-ako.sh ${i}
done

