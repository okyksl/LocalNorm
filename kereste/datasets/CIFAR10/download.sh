#!/bin/bash

mkdir $1

# Download Dataset
wget 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' -P $1

# Extract zips
tar -C $1 -zxvf $1/*.gz
rm $1/*.gz