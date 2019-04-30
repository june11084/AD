#!/bin/bash

for f in iot_data/freezer_data/processed/*.TRAIN
do 
  filename=$(basename -- "$f")
  extension="${filename##*.}"
	filename="${filename%.*}"
  echo $filename
  mkdir figs/$filename
  mkdir figs/$filename/alpha001
  python3 main_lstm.py iot_data/freezer_data/processed/$filename.TRAIN iot_data/freezer_data/processed/$filename.TEST --train test_student --custom_data iot_data/freezer_data/processed/$filename.with_std --load_check saved_models/$filename/lstm_student_h20_4layer_alpha001.pk --analysis 1 --alpha 0.01 --fig_path figs/$filename/alpha001/
done
