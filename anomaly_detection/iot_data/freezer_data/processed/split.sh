#!/bin/bash

FILES=*
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
	split -l 5000 $f
  mv xaa $f.TRAIN
  mv xab $f.TEST
done

