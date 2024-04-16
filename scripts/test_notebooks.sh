#!/bin/bash

pip install ipython
pip install nbformat
pip install seaborn
for nb in notebooks/*.ipynb; do
	echo "Running notebook smoke test on $nb"
	ipython -c "%run $nb" || exit 1
done
