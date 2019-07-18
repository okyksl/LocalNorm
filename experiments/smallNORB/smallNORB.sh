#!/bin/bash

python 'run.py' --experiment 'experiments/smallNORB/lighting/lighting.json'
python 'run.py' --experiment 'experiments/smallNORB/elevation/elevation.json'
python 'run.py' --experiment 'experiments/smallNORB/azimuth/azimuth.json'
python 'run.py' --experiment 'experiments/smallNORB/adversarial/adversarial.json'
python 'run.py' --experiment 'experiments/smallNORB/mixed/mixed.json'
