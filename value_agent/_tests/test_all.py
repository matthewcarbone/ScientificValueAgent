from pathlib import Path
import pytest

from yaml import safe_load

from value_agent.experiments import execute


@pytest.mark.parametrize(
    "yaml_file",
    [
        "job_xrd_1d_asymmetric_vf.yaml",
        "job_sine_2d_asymmetric_vf.yaml",
        "job_xrd4phase_2d_symmetric_vf.yaml",
        "job_xrd4phase_2d_asymmetric_vf.yaml",
        "job_sine_2d_symmetric_vf.yaml",
    ],
)
def test_run(yaml_file):
    yaml_file = Path("value_agent") / "_tests" / yaml_file
    params = safe_load(open(yaml_file, "r"))
    execute(params)

    # Later: might want to load in some of these json files and do some tests
    # on them
    for p in Path(".").glob("*seed*.json"):
        p.unlink()
