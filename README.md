# Level Set Teleportation: An Optimization Perspective 

Code to replicate experiments in the paper 
[Level Set Teleportation: An Optimization Perspective](https://arxiv.org/abs/2403.03362) by Aaron 
Mishkin, Alberto Bietti, and Robert M. Gower.

### Requirements

Python 3.8 or newer.

### Setup

Clone the repository using

```
git clone https://github.com/aaronpmishkin/teleport.git
```

We provide a script for easy setup on Unix systems. Run the `setup.sh` file with

```
./setup.sh
```

This will:

1. Create a virtual environment in `.venv` and install the project dependencies.
2. Install our modified version of [`stepback`](https://github.com/fabian-sp/step-back) in development mode. 
This library contains infrastructure for running our experiments.
3. Create the `data`, `figures`, and `results`  directories.

After running `setup.sh`, you need to activate the virtualenv using

```
source .venv/bin/activate
```

### Replications

The experiments are run via a command-line interface.
First, make sure that the virtual environment is active.
Running `which python` in bash will show you where the active Python binaries are; 
this will point to a file in `teleport/.venv/bin` if the virtual 
environment is active.
Then, execute one of the files in the `scripts/` directory. 
Before running the experiments, you must compute the optimal values 
for each of the UCI datasets with which we experiment. 
You can do this by running,
```
python scripts/run.py -E uci_optimal_value
```
Then, run `python scripts/extract_optimal_values.py` to extract the optimal
values into a pickle file.
The rest of the experiments can be run using the same command line interface
with the name of the desired experiment.
For example, the Newton comparison experiments in Figures 6, 15, and 16 can
be run using
```
python scripts/run.py -E figures_6_15_16.py
```
You can look at the experiment configuration files in `scripts/exp_configs`
to see the experiment names.

There are two ways to regenerate the plots from the paper.
Some plots are generated using a similar command-line interface to the experiments.
For example, to generate Figures 15 and 16, you can execute,
```
python scripts/plot.py -E figures_6_15_16 -P figures_15_16
```
where the `-E figures_6_15_16` specifies which experiment to load the data
from and `figures_15_16` specifies the plotting configuration.
You can look at the plot configuration files in `scripts/plot_configs` to
see the plot names.
Other figures, like Figure 6, are generated directly using custom scripts. 
To make Figure 6, you can run,
```
python scripts/make_figure_6.py
```
Finally, Figure 1 was created using a different library which has been included
in a sub-directory called `figure_1_code`.
To run the experiments and generate Figure 1 with a single command, run
```
python figure_1_code/make_figure_1.py 
```

Note that the UCI datasets must be manually retrieved from 
[here](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz).


### Citation

Please cite our paper if you make use of our code or figures from our paper. 

### Bugs or Other Issues

Please open an issue if you experience any bugs or have trouble replicating the experiments.

### Acknowledgements

The plotting code for Figure 1 is based [this repository](https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective).
