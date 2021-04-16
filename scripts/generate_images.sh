#!/bin/bash
echo "Model name: $1"
echo "Grid size: $2"
echo "LSV size: $3"
echo "Num runs: $4"
for run in `seq 1 $4`
do
  python generate_images.py --model_file_name=$1 --grid_size=$2 --lsv_size=$3
done

