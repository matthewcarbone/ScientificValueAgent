#!/bin/bash

bash scripts/install.sh build
echo "__version__ = '$(dunamai from any --style=pep440 --no-metadata)'" >sva/_version.py
flit build
git checkout sva/_version.py
