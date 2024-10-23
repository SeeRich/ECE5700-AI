#! /bin/bash

#### Convert ipynb to pdf
filename=$1
# extension="${filename##*.}"
filename="${filename%.*}"

# HACK to remove widget state first
jq -M 'del(.metadata.widgets)' $1 > $filename.temp

# Convert to html and then pdf
jupyter nbconvert --to html $filename.temp
jupyter nbconvert --to pdf $filename.temp

rm $filename.temp