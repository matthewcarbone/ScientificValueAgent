set dotenv-load

print-version:
    uvx --with hatch hatch version

version *VERSION: print-version
    uvx --with hatch hatch version {{ VERSION }}
    sed -n "s/__version__ = '\(.*\)'/\\1/p" "$PACKAGE_NAME"/_version.py > .version.tmp
    git add "$PACKAGE_NAME"/_version.py
    uv lock --upgrade-package "$PACKAGE_NAME"
    git add uv.lockfile
    git commit -m "Bump version to $(cat .version.tmp)"
    git tag -a "v$(cat .version.tmp)" -m "Bump version to $(cat .version.tmp)"
    rm .version.tmp

serve-jupyter:
    uv run --extra notebook --extra utilities jupyter lab --notebook-dir="~"

run-ipython:
    uv run ipython

