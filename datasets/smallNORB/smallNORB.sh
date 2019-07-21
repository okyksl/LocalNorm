#!/bin/bash

mkdir datasets/smallNORB/download

# Download Dataset
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz' -P datasets/smallNORB/download
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz' -P datasets/smallNORB/download
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz' -P datasets/smallNORB/download
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz' -P datasets/smallNORB/download
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz' -P datasets/smallNORB/download
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz' -P datasets/smallNORB/download

# Extract zips
gunzip datasets/smallNORB/download/*.gz
rm datasets/smallNORB/download/*.gz

# Download Dataset Parser
git clone https://github.com/okyksl/small_norb datasets/smallNORB/small_norb