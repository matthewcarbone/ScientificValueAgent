<div align=center>

# The Scientific Value Agent

[![image](https://github.com/matthewcarbone/ScientificValueAgent/actions/workflows/ci.yaml/badge.svg)](https://github.com/matthewcarbone/ScientificValueAgent/actions/workflows/ci.yaml)

</div>

The Scientific Value Agent (SVA) is a collection of tools for Gaussian Process (GP)-based Bayesian Optimization, particularly for use in epistemic experimentation (experiments in which you know nothing, have nothing to optimize and want to simply explore the experimental space as completely as you can).

> [!NOTE]
> If you use the Scientific Value Agent, please consider citing our [manuscript](https://doi.org/10.1016/j.matt.2023.11.012) in Matter. The software corresponding to this manuscript is tagged as [`v1.1.0`](https://github.com/matthewcarbone/ScientificValueAgent/tree/v1.1.0).

# 🧱 Install

Currently, the best way to install the SVA code is to simply clone the repository, then using `uv` to run campaigns (which installs the package to a local environment in editable mode). First, [install uv](https://docs.astral.sh/uv/getting-started/installation/). Then,

```bash
git clone git@github.com:matthewcarbone/ScientificValueAgent.git
cd ScientificValueAgent
uv run sva_run experiment=Sine2Phase policy=sva_importance_sampling
```

You can also use `uv run sva_run -h` to print the options to the command line (powered by [Hydra](https://hydra.cc/docs/intro/)).

In addition, there is a `justfile` which has a few useful commands. First, install [just](https://github.com/casey/just). Then, you can e.g. serve a Jupyter notebook using the SVA environment with `just serve-jupyter`.


# 🚀 Examples

See the [scripts](https://github.com/matthewcarbone/ScientificValueAgent/tree/master/scripts) directory for some examples of how to use SVA on some pre-defined [experiments](https://github.com/matthewcarbone/ScientificValueAgent/tree/master/sva/experiments). These use [Task Spooler](https://github.com/justanhduc/task-spooler) to launch many jobs in parallel, but you can simply remove the `ts` prefix to the command to launch them in your shell.

Once jobs have completed, [`read_data`](https://github.com/matthewcarbone/ScientificValueAgent/blob/41863beb95b9dc31bdf29e918b4ed3ec969dd1b5/sva/postprocessing.py#L42) can be used from `sva.postprocessing` to load all results, which can be further analyzed in e.g. a Jupyter Notebook.

# 💲 Funding acknowledgement

This software is supported by Brookhaven National Laboratory (BNL), Laboratory Directed Research and Development (LDRD) Grants: 

- No. 22-059, "Precision synthesis of multiscale nanomaterials through AI-guided robotics for advanced catalysts”
- No. 23-039, "Extensible robotic beamline scientist for self-driving total scattering studies"
- No. 24-004, "Human-AI-facility integration for the multi-modal studies on high-entropy nanoparticles"

This software was also supported by the AFRL Regional Network Mid-Atlantic under Cooperative Agreement FA8750-22-2-0500 with the Air Force Research Laboratory.

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.
