#!/bin/bash

# Change directory
source /home/dani/miniconda3/etc/profile.d/conda.sh

# Select the environment where you have installed the required libraries
conda activate tf 

# Execute the main program
python main.py