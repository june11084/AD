#!/bin/bash

for f in iot_data/freezer_data/processed/*.TRAIN
do 
  filename=$(basename -- "$f")
  extension="${filename##*.}"
	filename="${filename%.*}"
  echo $filename
  mkdir saved_models/$filename
  mkdir saved_models/$filename/vae
	mkdir saved_results/$filename/vae
  python3 main_vae.py iot_data/freezer_data/processed/$filename.TRAIN iot_data/freezer_data/processed/$filename.TEST --check_path saved_models/$filename/vae_h5.pk --results_path saved_results/$filename/vae/
  # python3 main_vae.py iot_data/freezer_data/processed/$filename.TRAIN iot_data/freezer_data/processed/$filename.TEST --load_check saved_models/$filename/vae_h5.pk
done
