apply-version:
    echo "__version__ = '$(uvx --with dunamai dunamai from any)'" > sva/_version.py
    cat sva/_version.py
    git add sva/_version.py
serve-jupyter:
    uv run --extra notebook --extra utilities jupyter lab --notebook-dir="~"
