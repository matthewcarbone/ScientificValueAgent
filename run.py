import yaml

from value_agent.experiments import execute


if __name__ == "__main__":
    params = yaml.safe_load(open("jobs.yaml", "r"))
    execute(params)
