#!/bin/bash
echo "Model file name: $1"
echo "Image path: $2"
echo "Num runs: $3"
for run in `seq 1 $3`
do
  python score_image.py --model_file_name=$1 --image_path=$2
done

