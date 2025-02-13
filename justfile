set dotenv-load

print-version:
    @echo "Current version is:" `uvx --with hatch hatch version`

[confirm]
apply-version *VERSION: print-version
    uvx --with hatch hatch version {{ VERSION }}
    sed -n "s/__version__ = '\(.*\)'/\\1/p" "$PACKAGE_NAME"/_version.py > .version.tmp
    git add "$PACKAGE_NAME"/_version.py
    uv lock --upgrade-package "$PACKAGE_NAME"
    git add uv.lock
    git commit -m "Bump version to $(cat .version.tmp)"
    if [ {{ VERSION }} != "dev" ]; then git tag -a "v$(cat .version.tmp)" -m "Bump version to $(cat .version.tmp)"; fi
    rm .version.tmp

serve-jupyter:
    uv run --with=ipython,jupyterlab,matplotlib,seaborn,h5netcdf,netcdf4,scikit-learn,scipy,xarray,"nbconvert==5.6.1" jupyter lab --notebook-dir="~"

serve-jupyter-no-browser:
    pm2 start "uv run --with=ipython,jupyterlab,matplotlib,seaborn,h5netcdf,netcdf4,scikit-learn,scipy,xarray,imageio,"nbconvert==5.6.1" jupyter lab --notebook-dir="~" --no-browser" --name jupyter-sva

run-ipython:
    uv run ipython

