#!/usr/bin/env bash

export PATH_HOME=$PWD

#for path in 01 02 03 04 05 06 07 08 09 12; do
for path in 1 2 3 4 5 6 7 8; do
#for path in 12; do
#  echo ${path}*
  cd ex${path}*
  innerpath=$PWD
#  echo "currentPath=${innerpath}"

  jupyter nbconvert *ipynb --to python
  cd ..
done

