# Scientific Value Agent

The ScientificValueAgent is a collection of tools for GP-based Bayesian Optimization, particularly for use in epistemic experimentation (experiments in which you know nothing, have nothing to optimize and want to simply explore the experimental space as completely as you can).

> [!NOTE]
> If you use the Scientific Value Agent, please consider citing our [manuscript](https://doi.org/10.1016/j.matt.2023.11.012) in Matter. The software corresponding to this manuscript is tagged as [`v1.1.0`](https://github.com/matthewcarbone/ScientificValueAgent/releases/tag/v1.1.0).
> Note that this code is currently only usable with CPU. Make sure you run
> ```bash
> export CUDA_VISIBLE_DEVICES=""
> ```
> or something similar before doing any training.

## Funding acknowledgement

This software is supported by Brookhaven National Laboratory (BNL), Laboratory Directed Research and Development (LDRD) Grants: No. 22-059, "Precision synthesis of multiscale nanomaterials through AI-guided robotics for advanced catalysts,” No. 23-039, "Extensible robotic beamline scientist for self-driving total scattering studies" and No. 24-004, "Human-AI-facility integration for the multi-modal studies on high-entropy nanoparticles."

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.
