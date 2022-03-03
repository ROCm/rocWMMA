#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build doxygen info
bash run_doxygen.sh

#install sphinx_rtd_theme using pip3
#install latexmk - download the latexmk archive from CTAN (https://ctan.org/pkg/latexmk?lang=en), and
#copy latexmk.pl file location to $PATH

# Build sphinx docs
cd source
make clean
make html
make latexpdf
