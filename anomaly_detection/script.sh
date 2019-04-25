#!/bin/bash

for f in iot_data/freezer_data/processed/*.TRAIN
do 
  filename=$(basename -- "$f")
  extension="${filename##*.}"
	filename="${filename%.*}"
  echo $filename
  mkdir saved_models/$filename
  python3 main_lstm.py iot_data/freezer_data/processed/$filename.TRAIN iot_data/freezer_data/processed/$filename.TEST --train train_teacher --check_path saved_models/$filename/lstm_teacher_h20_4layer.pk
	python3 main_lstm.py iot_data/freezer_data/processed/$filename.TRAIN iot_data/freezer_data/processed/$filename.TEST --build_std 1 --custom_data iot_data/freezer_data/processed/$filename.with_std --load_check saved_models/$filename/lstm_teacher_h20_4layer.pk
  python3 main_lstm.py iot_data/freezer_data/processed/$filename.TRAIN iot_data/freezer_data/processed/$filename.TEST --train_student --custom_data iot_data/freezer_data/processed/$filename.with_std --load_check saved_models/$filename/lstm_teacher_h20_4layer.pk --check_path saved_models/$filename/lstm_student_h20_4layer_alpha001.pk
done
