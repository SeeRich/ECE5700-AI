#! /bin/bash

# Convert ipynb to pdf
jupyter nbconvert --to html $1
jupyter nbconvert --to qtpdf $1