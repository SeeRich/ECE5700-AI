#! /bin/bash

# Convert ipynb to pdf
# HACK to remove widget state first
jq -M 'del(.metadata.widgets)' $1 > $1.temp
jupyter nbconvert --to html $1.temp
jupyter nbconvert --to qtpdf $1.temp
rm $1.temp