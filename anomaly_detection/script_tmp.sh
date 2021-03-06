#!/bin/bash

for f in iot_data/freezer_data/processed/*.TRAIN
do 
  filename=$(basename -- "$f")
  extension="${filename##*.}"
	filename="${filename%.*}"
  echo $filename
  mkdir saved_results/$filename
  mkdir saved_results/$filename/lstm_alpha01
  python3 main_lstm.py iot_data/freezer_data/processed/$filename.TRAIN iot_data/freezer_data/processed/$filename.TEST --train test_student --custom_data iot_data/freezer_data/processed/$filename.with_std --load_check saved_models/$filename/lstm_student_h20_4layer_alpha01.pk --analysis 1 --alpha 0.1 --results_path saved_results/$filename/lstm_alpha01/
done
