#!/usr/bin/env bash

export PATH_HOME=$PWD

# find . -type f -follow -print|xargs ls -LR | grep ipynb | grep -v .ipynb_ | awk '{ print $9 }'
# list file with space: for f in *\ *; do echo "$f" ; done
# remove space in file names: for f in *\ *; do mv "$f" "${f// /_}"; done
# for f in *\ *; do mv "$f" "${f//-/}" ; done
# for f in *\ *; do mv "$f" "${f// /_}" ; done

#for line in "$(find . -type f | grep ipynb | grep -v .ipynb_ | awk '{ print $9 }')"
#
##for path in 01 02 03 04 05 06 07 08 09 12; do
input="input2.txt"
while IFS= read -r line
do
  echo "${line}"
  jupyter nbconvert "${line}" --to python

done < "$input"

#function trspace() {
#   declare dir name bname dname newname replace_char
#   [ $# -lt 1 -o $# -gt 2 ] && { echo "usage: trspace dir char"; return 1; }
#   dir="${1}"
#   replace_char="${2:-_}"
#   find "${dir}" -xdev -depth -name $'*[ \t\r\n\v\f]*' -exec bash -c '
#      for ((i=1; i<=$#; i++)); do
#         name="${@:i:1}"
#         dname="${name%/*}"
#         bname="${name##*/}"
#         newname="${dname}/${bname//[[:space:]]/${0}}"
#         if [[ -e "${newname}" ]]; then
#            echo "Warning: file already exists: ${newname}" 1>&2
#         else
#            mv "${name}" "${newname}"
#         fi
#      done
#  ' "${replace_char}" '{}' +
#}
#
#trspace . _