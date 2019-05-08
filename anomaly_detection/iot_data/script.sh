#!/bin/bash

FILES=freezer_data/raw/*
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
	python3 read_data.py $f freezer_data/processed
done

