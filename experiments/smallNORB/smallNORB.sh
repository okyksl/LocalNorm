#!/bin/bash

python '../../run.py' --experiment 'lighting/lighting.json'
python '../../run.py' --experiment 'elevation/elevation.json'
python '../../run.py' --experiment 'azimuth/azimuth.json'
python '../../run.py' --experiment 'adversarial/adversarial.json'
python '../../run.py' --experiment 'mixed/mixed.json'
