# Economic nonlinear predictive control of water distribution networks based on surrogate modeling and automatic clustering
Felix Fiedler, Andrea Cominola, Sergio Lucia. TU Berlin and Einstein Center Digital Future, 2019.

##  Abstract
The operation of large-scale water distribution networks (WDNs) is a complex control task due to the size of the problem, the need to consider key operational, quality and safety-related constraints as well as because of the presence of uncertainties. An efficient operation of WDNs can lead to considerable reduction in the energy used to distribute the required amounts of water, leading to significant economic savings. Many model predictive control (MPC) schemes have been proposed in the literature to tackle this control problem. However, finding a control-oriented model that can be used in an optimization framework, which captures nonlinear behavior of the water network and is of a manageable size is a very important challenge faced in practice. We propose the use of a data-based automatic clustering method that clusters similar nodes of the network to reduce the model size and then learn a deep-learning based model of the clustered network. The learned model is used within an economic nonlinear MPC framework. The proposed method leads to a flexible scheme for economic robust nonlinear MPC of large WDNs that can be solved in real time, leads to significant energy savings and is robust to uncertain water demands. The potential of the proposed approach is illustrated by simulation results of a benchmark WDN model.

## About this code
This is the code base for the publication **Economic nonlinear predictive control of water distribution networks based on surrogate modeling and automatic clustering** (submitted: IFAC 2020). The results shown in this paper were exclusively created with the tools and files which are included in this repository. The only exception is that we do not supply the data base of simulation results used for training the surrogate model (due to size restrictions). This database, however, can be created with the different variants of `run_simulation_[...].py` in the `WNTR_Model` directory.

## About this repository
Note that we used Git LFS in this repository. Please follow the instructions [here](https://git-lfs.github.com/) to properly clone the LFS files.

**Please note that github can natively display jupyter notebooks. We extensively used annotations in these notebooks alongside with graphics and result snippets to illustrate our workflow. We suggest to visitors to investigate these notebooks on github to get a first impression of the repository. Please follow the instructions below on where to start.**

# Getting started
When investigating the codebase and this project we advise to follow the same structure as presented in the paper. 

## Clustering
The clustering directory contains the files
- wn_clustering.ipynb
- wn_clustering_plot.ipynb
The ipython notebook `wn_clustering` contains both the code and a comprehensive guide and evaluation for the applied clustering method. Given the database mentioned above, the resulting clustering for the subsequent steps is exported here and written to `.json` files.

## Surrogate_Model
This directory contains the files and tools used to create the deep learning surrogate model. The directory has the following structure (with only important files mentioned below):
- DNN
  - dnn_surrogate_prototyping.ipynb
  - dnn_surrogate_full_model.ipynb
  - surrogate_model_training_data.py
- RNN
  - RNN_surrogate_model.ipynb
  - RNN_tools.py
We investigated dense neural networks (DNN) and recurrent neural networks (RNN) as two very different model architectures for the surrogate model. 
Eventually, only DNN were used for the control task, due to their simplicity (in comparison) and their sufficient performance. When investigating this directory, we suggest to **work through the files in exactly the order stated above**.

In the file `dnn_surrogate_prototyping.ipynb` we investigated in depth the required data pre-processing pipeline and setup of the applied deep learning toolbox (Keras). In the spirit of rapid prototyping, this was conducted for a limited database only. After having validated the process the pre-processing algorithm was outsourced to the file `surrogate_model_training_data.py`. We then used a complete database within `dnn_surrogate_full_model.ipynb` to create the final model for the control task.

In the ipython notebook `RNN_surrogate_model.ipynb` we describe in detail, how to setup a recurrent neural network to use it as a surrogate model. The process also depends on `surrogate_model_training_data.py`, since data pre-processing is conducted identically. 

## MPC
For the control task, we use non-linear economic model predictive control (MPC) based on [CasADi](https://web.casadi.org/) for fast and efficient optimization. The subfolder `go_mpc` is organized as follows (with only important files mentioned below):
- mpc_backend.py
- mpc_main_loop.py
- mpc_results_anim.py

The optimization problem is setup in the file `mpc_backend.py`. Here, we also load the weights and configuration from the trained neural network and convert it into a CasADi symbolic expression. 
The MPC main loop runs in `mpc_main_loop.py`. In this file, we load the WNTR model and in a loop run the optimization, control the system and measure the state. Note that WNTR with Epanet simulator currently does not properly support starting and stopping. Thus, to create feedback at each time instance, we write the current control input to an Epanet configuration file (`.inp`) with the current timestep. This file gets updated at each iteration. To obtain the next state we re-simulate the entire sequence (from the intial condition) with the recorded sequence of optimal control inputs. Unfortunately, this means that the simulation time grows linearly over time. We found, however, that even for durations > 1 month (with timestep 1h) the simulation time is less than the optimization time (1-5 s). 

## WNTR_Model
This directory containts all the files and tools related to the WNTR model. Most importantly, we used `run_simulation_1stepControl.py` to create the training database for the surrogate model.
