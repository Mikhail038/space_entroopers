#!/bin/bash

make -C ./src/
./src/entropy_analyzer "$1" -o data.csv
python ./src/entropy_csv_visualizer.py data.csv
