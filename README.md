<h2 align='center'>torchssm</h2>
<h3 align='center'>Neural State-Space Models and Latent Dynamic Functions <br> for High-Dimensional Generative Time-series modeling</h3>

<a name="about"></a>
## About this Repository

This repository is meant to conceptually introduce and highlight implementation considerations for the recent class of models called <b>Neural State-Space Models (Neural SSMs)</b>. They leverage the classic state-space model with the flexibility of deep learning to approach high-dimensional generative time-series modeling and learning latent dynamics functions.

Included is an abstract PyTorch-Lightning training class with several latent dynamic functions that inherit it, as well as common metrics used in their evaluation and training examples on common datasets. Further broken down via implementation is the distinction between <i>system identification</i> and <i>state estimation</i> approaches, which are reminiscent of their classic SSM counterparts and arise from fundamental differences in the underlying choice of their probabilistic graphical model (PGM).

<a name="pgmSchematic"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169753112-bc849b24-fe13-4975-8697-fea95bb19fb5.png" alt="pgm schematic" /></p>
<p align='center'>Fig 1. Schematic of the two PGM forms of Neural SSMs.</p>


<!-- CITATION -->
<a name="citation"></a>
## Citation
If you found the information helpful for your work or use portions of this repo in research development, please consider citing:
```
@misc{missel2022torchssm,
    title={TorchSSM},
    author={Missel, Ryan},
    publisher={Github},
    journal={Github repository},
    howpublished={\url{https://github.com/qu-gg/torchssm}},
    year={2022},
}
```

<a name="toc"></a>
## Table of Contents
- [About](#about)
- [Citation](#citation)
- [Table of Contents](#toc)
- [Background](#background)
  - [What are Neural SSMs?](#neuralSSMwhat)
  - [Choice of SSM PGM](#pgmChoice)
  - [Latent Initial State Z<sub>0</sub> / Z<sub>init</sub>](#initialState)
  - [System Controls, u<sub>t</sub>](#ssmControls)
- [Implementation](#implementation)
  - [Datasets](#data)
  - [Models](#models)
  - [Metrics](#metrics)
- [Experiments](#experiments)
  - [Hyperparameter Tuning](#hyperparameters)
  - [Hamiltionian Systems](#hamiltonian)
- [Miscellaneous](#misc)
  - [To-Do](#todo)
  - [Contributions](#contributions)
  - [References](#references)

<!-- BACKGROUND -->
<a name="background"></a>
# Background

This section provides an introduction to the concept of Neural SSMs, some common considerations and limitations, and active areas of research. This section assumes some familiarity with state-space models, though little background is needed to gain a conceptual understanding if one is already coming from a latent modeling perspective or Bayesian learning. Resources are available in abundance considering the width and depth of state-space usage, however, this <a href="https://www.youtube.com/watch?v=hpeKrMG-WP0">video</a> and <a href="https://github.com/probml/ssm-book">modern textbook</a> are good starting points.

<!-- Neural SSM INTRO -->
<a name="neuralSSMwhat"></a>
## What are Neural SSMs?
An extension of classic state-space models, <i>neural</i> state-space models - at their core - consist of a dynamic function of some latent states <b>z_k</b> and their emission to observations <b>x_k</b>, realized through the equations:
<a name="ssmEQ"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169743189-057f52a5-8a08-4616-9516-3c60aca86b28.png" alt="neural ssm equations" )/></p>
where <b>θ</b><sub>z</sub> represents the parameters of the latent dynamic function. The precise form of these functions can vary significantly - from deterministic or stochastic, linear or non-linear, and discrete or continuous.
<p> </p>
Due to their explicit differentiation of transition and emission and leveraging of structured equations, they have found success in learning interpretable latent dynamic spaces<sup>[1,2,3]</sup>, identifying physical systems from non-direct features<sup>[4,5,6]</sup>, and uses in counterfactual forecasting<sup>[7,8,14]</sup>.
<p> </p>
Given the fast pace of progress in latent dynamics modeling over recent years, many models have been presented under a variety of terminologies and proposed frameworks - examples being variational latent recurrent models<sup>[5,9,10,11,12,22]</sup>, deep state-space models<sup>[1,2,3,7,13,14]</sup>, and deterministic encoding-decoding models<sup>[4,15,16]</sup>. Despite differences in appearance, they all adhere to the same conceptual framework of latent variable modeling and state-space disentanglement. As such, here we unify them under the terminology of Neural SSMs and segment them into the two base choices of probabilistic graphical models that they adhere to: <i>system identification</i> and <i>state estimation</i>. We highlight each PGM's properties and limitations with experimental evaluations on benchmark datasets.


<!-- PGM CHOICES -->
<a name="pgmChoice"></a>
## Choice of PGM - System Identification vs State Estimation
The PGM associated with each approach is determined by the latent variable chosen for inference.

<a name="latentSchematic"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172081773-d6af11d9-9c1e-4b04-8c65-a83e0fa80dbc.png" alt="latent variable schematic" /></p>
<p align='center'>Fig 2. Schematic of latent variable PGMs in Neural SSMS<sup>[MetaLearning]</sup>.</p>


<b>System states as latent variables (State Estimation)</b>: The intuitive choice for the latent variable is the latent state <b>z_k</b> that underlies <b>x_k</b>, given that it is already latent in the system and is directly associated with the observations. The PGM of this form is shown under Fig. [1A](#pgmSchematic) where its marginal likelihood over an observed sequence x<sub>0:T</sub> is written as:

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172077627-bc72445e-11d9-4344-86fc-0abbd0c723df.png" alt="state likelihood" /></p>

where <i>p</i>(<b>x</b><sub>i</sub> | <b>z</b><sub>i</sub>) describes the emission model and <i>p</i>(<b>z</b><sub>i</sub> | <b>z</b><sub><i</sub>, <b>x</b><sub><u><</u>i</sub>) describes the latent dynamics function. Given the common intractability of the posterior, parameter inference is performed through a variational approximation of the posterior density <i>q</i>(<b>z</b><sub>0:T</sub> | <b>x</b><sub>0:T</sub>), expressed as:

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172077640-16e26f56-bed3-4dec-b078-783e247eeda3.png" alt="state variational posterior" /></p>

Given these two components, the standard training objective of the Evidence Lower Bound Objective (ELBO) is thus derived with the form:

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172077692-452892fc-e5e6-4f4a-9296-e975130da816.png" alt="state ELBO" /></p>

where the first term represents a reconstruction likelihood term over the sequence and the second is a Kullback-Leibler divergence loss between the variational posterior approximation and some assumed prior of the latent dynamics. This prior can come in many forms, either being the standard Gaussian Normal in variational inference, flow-based priors from ODE settings<sup>[5]</sup>, or physics-based priors in problem-specific situations<sup>[20]</sup>. This is the primary design choice that separates current works in this area, specifically the modeling of the dynamics prior and its learned approximation. Many works draw inspiration for modeling this interaction by filtering techniques in standard SSMs, where a divergence term is constructed between the dynamics-predicted latent state and the data-corrected observation<sup>[7,18]</sup>.

With this formulation, it is easy to see how dynamics models of this type can have a strong reconstructive capacity for the high-dimensional outputs and contain strong short-term predictions. In addition, input-influenced dynamics are inherent to the prediction task, as errors in the predictions of the latent dynamics are corrected by true observations every step. However, given this data-based correction, the resulting inference of <i>q</i>(<b>z</b><sub>i</sub> | <b>z</b><sub><i</sub>, <b>x</b><sub><u><</u>i</sub>) is weakened, and without near-term observations to guide the dynamics function, its long-horizon forecasting is limited<sup>[1,3]</sup>.

<b>System parameters as latent variables (System Identification)</b>: Rather than system states, one can instead choose to select the parameters (denoted as <b>θ</b><sub>z</sub> in Equation [1](#ssmEQ)). With this change, the resulting PGM is represented in Fig. [1B](#pgmSchematic) and its marginal likelihood over x<sub>0:T</sub> is represented now by:

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172077614-52b7ff61-24f6-4828-8714-a73b57628e41.png" alt="sysID likelihood" /></p>

where the resulting output observations are derived from an initial latent state <b>z</b><sub>0</sub> and the dynamics parameters <b>θ</b><sub>z</sub>. As before, a variational approximation is considered for inference in place of an intractable posterior but now for the density <i>q</i>(<b>θ</b><sub>z</sub>, <b>z</b><sub>0</sub>) instead. Given prior density assumptions of  <i>p</i>(<b>θ</b><sub>z</sub>) and <i>p</i>(<b>z</b><sub>0</sub>) in a similar vein as above, the ELBO function for this PGM is constructed as:

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172078432-50f30273-508a-4286-98a4-bdd3fa03bc03.png" alt="sysID ELBO" /></p>
where again the first term is a reconstruction likelihood and the terms following represent KL-Divergence losses over the inferred variables.

<p> </p>
The given formulation here is the most general form for this line of models and other works can be covered under special assumptions or restrictions of how <i>q</i>(<b>θ</b><sub>z</sub>) and <i>p</i>(<b>θ</b><sub>z</sub>) are modeled. Original Neural SSM parameter works consider Linear-Gaussian SSMs as the transition function and introduce non-linearity by varying the transition parameters over time as <b>θ</b><sub>z<sub>0:T</sub></sub><sup>[1,2,3]</sup>. However, as shown in Fig. [2B<sub>1</sub>](#latentSchematic), the result of this results in convoluted temporal modeling and devolves into the same state estimation problem as now the time-varying parameters rely on near-term observations for correctness<sup>[8,20]</sup>. Rather than time-varying, the system parameters could be considered an optimized global variable, in which the underlying dynamics function becomes a Bayesian neural network in a VAE's latent space<sup>[5]</sup> and is shown in Fig. [2B<sub>2</sub>](#latentSchematic). Restricting these parameters to be deterministic results in a model of the form presented in Latent ODE<sup>[10]</sup>. The furthest restriction in forgoing stochasticity in the inference of <b>z</b><sub>0</sub> results in the suite of models as presented in [4]. 

<p> </p>
Regardless of the precise assumptions, this framework builds a strong latent dynamics function that enables long-term forecasting and, in some settings, even full-scale system identification<sup>[1,4]</sup> of physical systems. This is done at the cost of a harder inference task given no access to dynamics correction during generation and for full identification tasks, often requires a large number of training samples over the potential system state space<sup>[4,5]</sup>.

<!-- INITIAL STATE -->
<a name="initialState"></a>
## Latent Initial State Z<sub>0</sub> / Z<sub>init</sub>

As the transition dynamics and the observation space are intentionally disconnected in this framework, the problem of inferring a strong initial latent state from which to forecast is an important consideration when designing a neural state-space model. This is primarily a task- and data-dependent choice, in which the architecture follows the data structure. Thankfully, much work has been done in other research directions on designing good latent encoding models. As such, works in this area often draw from them. This section is split into three parts - one on the usual architecture for high-dimensional image tasks, one on lower-dimensional and/or miscellaneous encoders, and one on the difference between <b>z</b><sub>0</sub> and <b>z</b><sub>init</sub>.

<b>Image-based Encoders</b>: Unsurprisingly, the common architecture used in latent image encoding is a convolutional neural network (CNN) given its inherent bias toward spatial feature extraction<sup>[1,3,4,5]</sup>. Works are mixed between either having the sequential input reshaped as frames stacked over the channel dimension or simply running the CNN over each observed frame separately and passing the concatenated embedding into an output layer. Regardless of methodology, a few frames are assumed as observations for initialization, as multiple timesteps are required to infer the initial system movement. A subset of works considers second-order latent vector spaces, in which the encoder is explicitly split into two individual position and momenta functions, taking single and multiple frames respectively<sup>[5]</sup>.

<a name="initialStateSchematic"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/173459329-c48b1f65-2b26-4287-96e9-575a4bc8c540.png", width=500, alt="initial state visualization" /></p>
<p align='center'>Fig N. Visualization of the stacked initial state encoder, modified from [23].</p>


<b>Alternate Encoders</b>: In settings with non-image-based inputs, the initial latent encoder can take on a large variety of forms, ranging anywhere from simple linear/MLP networks in physical systems<sup>[5]</sup> to graph convolution networks for latent medical image forecasting<sup>[20]</sup>. Multi-modal and dynamics conditioning inputs can be leveraged via combinations of encoders whose embeddings go through a shared linear function.

<a name="stackedGCNNEncoder"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/173444301-1d31210a-98a0-40f8-9f78-5db95bcf90c4.png", width=500, alt="alternate encoder visualization" /></p>
<p align='center'>Fig N. Visualization of the stacked graph convolutional encoder, modified from [24].</p>


<b>Variables <b>z</b><sub>0</sub> vs. <b>z</b><sub>init</sub></b>:
Beyond just the inference of this latent variable, there is one more variation that can be seen throughout literature - that of whether the inferred variable is directly <b>z</b><sub>0</sub> or a pre-forecast variable <b>z</b><sub>init</sub>. For <b>z</b><sub>0</sub>, the initial latent state directly goes through the decoder for the initial reconstruction <b>x</b><sub>0</sub> without any interaction with the dynamics function. On the other hand, <b>z</b><sub>init</sub> represents an abstract initial vector state that the dynamics function starts from to get to <b>z</b><sub>0</sub> and <b>x</b><sub>0</sub>. It is a subtle distinction but potentially has implications for the resulting likelihood optimization and learned vector space. 

<a name="z0vszinit"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/173461096-4b9611fc-6cb0-4b1c-820b-daa46b2335c1.png", width=500, alt="initial latent variable comparison" /></p>
<p align='center'>Fig N. Schematic of the difference between <b>z</b><sub>0</sub> and <b>z</b><sub>init</sub></b> formulations.

Saying that, however, there is not much work exploring the considerations for each approach, besides ad-hoc solutions to bridge the gap between the latent encoder and dynamics function distributions<sup>[5]</sup>. This gap can stem from optimization problems caused by imbalanced reconstruction terms between dynamics and initial states or in cases where the initial state distribution is far enough away from the data distribution of downstream frames. A variety of empirical techniques have been proposed to tackle this gap, much in the same spirit of empirical VAE stability 'tricks.' These include separated <b>x</b><sub>0</sub> and <b>x</b><sub>1:T</sub> terms (where <b>x</b><sub>0</sub> has a positive weighting coefficient), VAE pre-training for <b>x</b><sub>0</sub>, and KL-regularization terms between the output distributions of the encoder and the dynamics flow<sup>[1,5]</sup>. One <i>personal</i> intuition regarding these two variable approaches and the tricks applied is that there exists a theoretical trade-off between the two formulations and the tricks applied help to empirically alleviate the shortcomings of either approach. This, however, requires experimentation and validation before any claims can be made.


<!-- CONTROLS -->
<a name="ssmControls"></a>
## Use of System Controls, u<sub>t</sub>

Insofar we have ignored another common and important component of state-space modeling, the incorporation of external controls <i>u</i> that affect the transition function of the state. Controls represent factors that influence the trajectory of a system but are not direct features of the object/system being modeled. For example, an external force such as friction acting on a moving ball or medications given to a patient could be considered controls<sup>[8,14]</sup>. These allow an additional layer of interpretability in SSMs and even enable counterfactual reasoning; i.e., given the current state, what does its trajectory look like under varying control inputs going forwards? This has myriad uses in medical modeling with counterfactual medicine<sup>[14]</sup> or physical system simulations<sup>[8]</sup>.
<p> </p>

For Neural SSMs, a variety of approaches have been taken thus far depending on the type of latent transition function used.
<p> </p>

<b>Linear Dynamics</b>: In latent dynamics still parameterized by traditional linear gaussian transition functions, control incorporation is as easy as the addition of another transition matrix <b>B<sub>t</sub></b> that modifies a control input <b>u<sub>t</sub></b> at each timestep<sup>[1,2,4,7]</sup>.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170075684-2ba31e45-b66f-4d3c-aed6-9ab28def95d6.png" alt="linear control" /></p>
<p align='center'>Fig N. Example of control input in a linear transition function<sup>[1]</sup>.</p>
<p> </p>

<b>Non-Linear Dynamics</b>: In discrete non-linear transition matrices using either multi-layer perceptrons or recurrent cells, these can be leveraged by either concatenating it to the input vector before the network forward pass or as a data transformation in the form of element-wise addition and a weighted combination<sup>[10]</sup>.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170173582-a8158240-62d0-4b7e-8793-d1c796bc4a6c.png" alt="non-linear control" /></p>
<p align='center'>Fig N. Example of control input in a non-linear transition function<sup>[1]</sup>.</p>
<p> </p>

<b>Continuous Dynamics</b>: For incorporation into continuous latent dynamics functions, finding the best approaches is an ongoing topic of interest. Thus far, the reigning approaches are:

1. Directly jumping the vector field state with recurrent cells<sup>[18]</sup>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170078493-b7d10d50-d252-4258-bed7-f7c2ae1080b9.png" alt="jump control" /></p>

2. Influencing the vector field gradient (e.g. neural controlled differential equations)<sup>[17]</sup>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170079172-b2dd6376-628d-4e15-8282-4ee296cd5b89.png" alt="gradient control" /></p>

3. Introducing another dynamics mechanism, continuous or otherwise (e.g. neural ODE or attention blocks), that is combined with the latent trajectory <b>z<sub>1:T</sub></b> into an auxiliary state <b>h<sub>1:T</sub></b><sup>[8,14]</sup>.
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170077468-f183e75f-3ad0-450e-b402-6718087c9b9c.png" alt="continuous control" /></p>
<p align='center'>Fig N. Visualization of the IMODE architecture, taken from [8].</p>


<!-- META-LEARNING -->
<!-- <a name="metaLearning"></a>
## Meta-Learning Dynamics -->


<!-- IMPLEMENTATION -->
<a name="implementation"></a>
# Implementation

In this section, specifics on model implementation and the datasets/metrics used are detailed. Specific data generation details are available in the URLs provided for each dataset. The models and datasets used throughout this repo are solely grayscale physics datasets with underlying Hamiltonian laws, such as pendulum and mass-spring sets. Extensions to color images and non-pixel-based tasks (or even graph-based data!) are easily done in this framework, as the only architecture change needed is the structure of the encoder and decoder networks as the state propagation happens solely in a latent space.

The project's folder structure is as follows:
<a name="folderStructure"></a>
```
  torchssm/
  │
  ├── train.py                      # Training entry point that takes in user args and handles boilerplate
  ├── README.md                     # What you're reading right now :^)
  ├── requirements.txt              # Anaconda requirements file to enable easy setup
  |
  ├── data/
  |   ├── <dataset_type>            # Name of the stored dynamics dataset (e.g. pendulum)
  |   ├── generate_bouncingball.py  # Dataset generation script for Bouncing Ball
  |   ├── generate_hamiltonian.py   # Dataset generation script for Hamiltonian Dynamics
  |   └── tar_directory.py          # WebDataset generation script 
  ├── experiments/
  |   └── <model_name>              # Name of the dynamics model run
  |       └── <experiment_type>     # Given name for the ran experiment
  |           └── <version_x>/      # Each experiment type has its sequential lightning logs saved
  ├── lightning_logs/
  |   ├── version_0/                # Placeholder lightning log folder
  |   └── ...                       # Subsequent runs
  ├── models/
  │   ├── CommonDynamics.py         # Abstract PyTorch-Lightning Module to handle train/test loops
  │   ├── CommonVAE.py              # Shared encoder/decoder Modules for the VAE portion
  │   ├── system_identification/ 
  │       └── ...                   # Specific System-Identification dynamics functions
  │   └── state_estimation/
  │       └── ...                   # Specific State-Estimation dynamics functions
  ├── utils/
  │   ├── layers.py                 # PyTorch Modules that represent general network layers
  │   ├── metrics.py                # Metric functions for evaluation
  │   ├── plotting.py               # Plotting functions for visualizatin
  |   └── utils.py                  # General utility functions (e.g. argparsing, experiment number tracking, etc)
  └──
```

<!-- DATA -->
<a name="data"></a>
## Data

All data used throughout these experiments are available for download <a href="">here</a> on Google Drive, in which they already come in their WebDataset forms. The total sizes of all sets are under a modest 2GB. However, feel free to generate your own sets using the provided data scripts!

<b>Hamiltonian Dynamics</b>: Provided for evaluation are a <a href="https://github.com/webdataset/webdataset">WebDataset</a> dataloader and generation scripts for DeepMind's Hamiltonian Dynamics
<a href="https://github.com/deepmind/dm_hamiltonian_dynamics_suite">suite</a><sup>[4]</sup>, a simulation library for 17 different physics datasets that have known underlying Hamiltonian dynamics.
It comes in the form of color image sequences of arbitrary length, coupled with the system's ground truth states (e.g., for pendulum, angular velocity and angle). It is well-benchmarked and customizable, making it a perfect testbed for latent dynamics function evaluation. For each setting, the physical parameters are tweakable alongside an optional friction coefficient to construct non-energy conserving systems. The location of focal points and the color of the objects are all individually tuneable, enabling mixed and complex visual datasets of varying latent dynamics.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/171246437-0a1ef292-f90c-4fb7-beb3-82a5e74bb549.gif" alt="pendulum examples" /></p>
<p align='center'>Fig N. Pendulum-Colors Examples</p>

For the base presented experiments of this dataset, we consider and evaluate grayscale versions of pendulum and mass-spring - which conveniently are just the sliced red channel of the original sets. Each set has `20000` training and `2000` testing trajectories sampled at `Δt = 0.05` intervals. Energy conservation is preserved without friction and we assume constant placement of focal points for simplicity. Note that the modification to color outputs in this framework is as simple as modifying the number of channels in the encoder and decoder.

<p> </p>
<b>Bouncing Balls</b>: Additionally, we provide a dataloader and generation scripts for the standard latent dynamics dataset of bouncing balls<sup>[1,2,5,7,8]</sup>, modified from the implementation in <a href="https://github.com/simonkamronn/kvae/tree/master/kvae/datasets">[1]</a>. It consists of a ball or multiple balls moving within a bounding box while being affected by potential external effects, e.g. gravitational forces<sup>[1,2,5]</sup>, pong<sup>[2]</sup>, and interventions<sup>[8]</sup>. The starting position, angle, and velocity of the ball(s) are sampled uniformly between a set range. It is generated with the <a href="https://github.com/viblo/pymunk">PyMunk</a> and <a href="https://www.pygame.org/news">PyGame</a> libraries. In this repository, we consider two sets - a simple set of one gravitational force and a mixed set of 4 gravitational forces in the cardinal directions with varying strengths. We similarly generate <code>20000</code> training and <code>2000</code> testing trajectories, however sample them at <code>Δt = 0.1</code> intervals.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/171948373-ad692ecc-bfac-49dd-86c4-137a2a5e4b73.gif" alt="bouncing ball examples" /></p>
<p align='center'>Fig N. Single Gravity Bouncing  Ball Example</p>


<p> </p>
Notably, this system is surprisingly difficult to successfully perform long-term generation on, especially in cases of mixed gravities or multiple objects. Most works only report on generation within 5-15 timesteps following a period of 3-5 observation timesteps<sup>[1,2,7]</sup> and farther timesteps show lost trajectories and/or incoherent reconstructions.

<p> </p>
<b>Mixing Physics</b>: So far in the literature, the majority of works only consider training Neural SSMs on one system of dynamics at a time - with the most variety lying in that of differing trajectory hyper-parameters. The ability to infer multiple dynamical systems under one model (or learn to output dynamical functions given system observations) and leverage similarities between the sets is an ongoing research pursuit - with applications of neural unit hypernetworks<sup>[ICML MetaPrior]</sup> and dynamics functions conditioned on sequences via meta-learning<sup>[NeurIPS MetaSSM]</sup> being the first dives into this.

<p> </p>
<b>Other Sets in Literature</b>: Outside of the previous sets, there are a plethora of other datasets that have been explored in relevant work. The popular task of human motion prediction in both the pose estimation and video generation setting has been considered via datasets <a href="http://vision.imar.ro/human3.6m/pami-h36m.pdf">Human3.6Mil</a>, <a href="http://mocap.cs.cmu.edu/">CMU Mocap</a>, and <a href="https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html">Weizzman-Action</a><sup>[5,19]</sup>, though proper experimentation into this area would require problem-specific architectures given the depth of the existing field. Past high-dimensionality and image-space, standard benchmark datasets in time-series forecasting have also been considered, including the <a href="https://github.com/Mcompetitions/M4-methods">M4</a>, <a href="https://github.com/zhouhaoyi/ETDataset">Electricity Transformer Temperature (ETT)</a>, and <a href="https://www.nasa.gov/intelligent-systems-division">the NASA Turbofan Degradation</a> set. Recent works have begun looking at medical applications in inverse image reconstruction and the incorporation of physics-inspired priors<sup>[20]</sup>. Regardless of the success of Neural SSMs on these tasks, the task-agnostic factor and principled structure of this framework make it a versatile and appealing option for generative time-series modeling.


<!-- MODELS -->
<a name="models"></a>    
## Models

Here, details on how the model implementation is structured and running experiments locally are given. As well, an overview of the abstract class implementation for a general Neural SSM and its types are explained.

### Implementation Structure
Provided within this repository is a PyTorch class structure in which an abstract PyTorch-Lightning Module is shared across all the given models, from which the specific VAE and dynamics functions inherit and override the relevant forward functions for training and evaluation. Swapping between dynamics functions and PGM type is as easy as passing in the model's name for arguments, e.g. `python3 train.py --model node`. As the implementation is provided in <a href="https://pytorch-lightning.readthedocs.io/en/latest/">PyTorch-Lightning</a>, an optimization and boilerplate library for PyTorch, it is recommended to be familiar at face-level.

<p> </p>
For every model run, a new lightning_logs/ version folder is created as well as a new experiment version under `experiments` related to the passed in naming arguments. Hyperparameters passed in for this run are both stored in the Tensorboard instance created as well as in the local file <code>hparams.yaml</code>. Default values and available options can be found in <code>utils/utils.py</code> or by running <code>python3 train.py -h</code>. During training and validation sequences, all of the metrics below are automatically tracked and saved into a Tensorboard instance which can be used to compare different model runs following. Every 5 epochs, reconstruction sequences against the ground truth for a set of samples are saved to the experiments folder. Currently, only one checkpoint is saved based on the last epoch ran rather than checkpoints based on the best validation score or at set epochs. Restarting training from a checkpoint or loading in a model for testing is done currently by the <code>lightning_logs/</code> ID, e.g. <code>python3 train.py --ckpt 51</code>.

<p> </p>
The implemented dynamics functions are each separated into their respective PGM groups, however, they can still share the same general classes. Each dynamics subclass has a <code>model_specific_loss</code> function that allows it to return additional loss values without interrupting the abstract flow. For example, this could be used in a flow-based prior that has additional KL terms over ODE flow density without needing to override the <code>training_step</code> function with a duplicate copy.

### Implemented Dynamics

<b>System Identification Models</b>: 

<p> </p>
For the system identification models, we provide a variety of dynamics functions that resemble the general and special cases detailed above, which are provided in Fig N. below. The most general version is that of the Bayesian Neural ODE, in which a neural ordinary differential equation<sup>[21]</sup> is sampled from a set of optimized distributional parameters and used as the latent dynamics function <code>z<sup>'</sup><sub>t</sub> = f<sub><i>p</i>(θ)</sub>(z<sub>s</sub>)</code><sup>[5]</sup>. A deterministic version of a standard Neural ODE is similarly provided, e.g. <code>z<sup>'</sup><sub>t</sub> = f<sub>θ</sub>(z<sub>s</sub>)</code><sup>[10,21]</sup>. Following that, two forms of a Recurrent Generative Network are provided, a residual version (RGN-Res) and a full-step version (RGN), that represent deterministic and discrete non-linear transition functions. RGN-Res is the equivalent of a Neural ODE using a fixed step Euler integrator while RGN is just a recurrent forward step function. 
Additionally, a representation of the time-varying Linear-Gaussian SSM transition dynamics<sup>[1,2]</sup> (LGSSM) is provided. And finally, a set of autoregressive models are considered (i.e. Recurrent neural networks, Long-Short Term Memory networks, Gated Recurrent Unit networks) as baselines. Their PyTorch Cell implementations are used and evaluated over the entire sequence, passing in the previously predicted state and observation as its inputs.

<p> </p>
Training for these models has one mode, that of taking in several observational frames to infer <b>z</b><sub>0</sub> and  then outputting a full sequence autonomously without access to subsequent observations. A likelihood function is compared over the full reconstructed sequence and optimized over. Testing and generation in this setting can be done out to any horizon easily and we provide small sample datasets of <code>1000</code> timesteps to evaluate out to long horizons.

<p> </p>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172108058-481009a0-41c9-449e-bc0f-7b7a45ecefe0.png", height=400, alt="sysID models" /></p>
<p align='center'>Fig N. Complete model schematic for system identification's implemented dynamics functions.</p>
 
<b>State Estimation Models</b>: 

<p> </p>
For the state estimation line, we provide a reimplementation of the classic Neural SSM work Deep Kalman Filter<sup>[7]</sup> alongside state estimation versions of the above, provided in Fig. N below. The DKF model modifies the standard Kalman Filter Gaussian transition function to incorporate non-linearity and expressivity by parameterizing the distribution parameters with neural networks <code>z<sub>t</sub>∼<i>N</i>(G(z<sub>t−1</sub>,∆<sub>t</sub>), S(z<sub>t−1</sub>,∆<sub>t</sub>))</code><sup>[7]</sup>. The autoregressive versions for this setting can be viewed as a reimplementation of the Variational Recurrent Neural Network (VRNN), one of the starting state estimation works in Neural SSMs<sup>[22]</sup>. For the latent correction step, we leverage a standard Gated Recurrent Unit (GRU) cell and the corrected latent state is what is passed to the decoder and likelihood function. Notably, there are two settings these models can be run under: <i>reconstruction</i> and <i>generation</i>. <i>Reconstruction</i> is used for training and incorporates ground truth observations to correct the latent state while <i>generation</i> is used to test the forecasting abilities of the model, both short- and long-term.

<p> </p>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/172186199-602a868b-77e4-44a2-b88d-64124c43afb9.png", height=400, alt="stateEst models" /></p>
<p align='center'>Fig N. Complete model schematic for state estimation's implemented dynamics functions.</p>


<!-- METRICS -->
<a name="metrics"></a>
## Metrics

<b> Mean Squared Error (MSE)</b>: A common metric used in video and image tasks where its use is in per-frame average over individual pixel error. While a multitude of papers solely uses plots of frame MSE over time as an evaluation metric, it is insufficient for comparison between models - especially in cases where the dataset contains a small object for reconstruction<sup>[4]</sup>. This is especially prominent in tasks of system identification where a model that fails to predict long-term may end up with a lower average MSE than a model that has better generation but is slightly off in its object placement.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169945139-ebcfc6e9-14d5-4a88-bc38-a9ed41ff5dfc.png" alt="mse equation" /></p>
<p align='center'>Fig N. Per-Frame MSE Equation.</p>

<b>Valid Prediction Time (VPT)</b>: Introduced in [4], the VPT metric is an advance on latent dynamics evaluation over pure pixel-based MSE metrics. For each prediction sequence, the per-pixel MSE is taken over the frames individually, and the minimum timestep in which the MSE surpasses a pre-defined epsilon is considered the 'valid prediction time.' The resulting mean number over the samples is often normalized over the total prediction timesteps to get a percentage of valid predictions.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169942657-5208afb5-6faf-4d47-b9a2-ef0a89a5fc9f.png" alt="vpt equation" /></p>
<p align='center'>Fig N. Per-Sequence VPT Equation.</p>

<b>Object Distance (DST)</b>: Another potential metric to support evaluation (useful in image-based physics forecasting tasks) is using the Euclidean distance between the estimated center of the predicted object and its ground truth center. Otsu's Thresholding <a href="https://en.wikipedia.org/wiki/Otsu%27s_method">method</a> can be applied to grayscale output images to get binary predictions of each pixel and then the average pixel location of all the "active" pixels can be calculated. This approach can help alleviate the prior MSE issues of metric imbalance as the maximum Euclidean error of a given image space can be applied to model predictions that fail to have any pixels over Otsu's threshold.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169954436-74d02fdc-0ab3-4d2e-b2b4-b595e35144a0.png" alt="dst equation" /></p>
<p align='center'>Fig N. Per-Frame DST Equation.</p>
where R<sup>N</sup> is the dimension of the output (e.g. number of image channels) and s, s<sub>hat</sub> are the subsets of "active" predicted pixels.

<p> </p>
<b>Valid Prediction Distance (VPD)</b>: Similar in spirit to how VPT leverages MSE, VPD is the minimum timestep in which the DST metric surpasses a pre-defined epsilon. This is useful in tracking how long a model can generate an object in a physical system before either incorrect trajectories and/or error accumulation cause significant divergence.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169961714-2d007dbd-92f2-4ec7-aff0-3c383f21e919.png" alt="vpd equation" /></p>
<p align='center'>Fig N. Per-Sequence VPD Equation.</p>

<p> </p>
<b>R<sup>2</sup> Score</b>: For evaluating systems where the full underlying latent system is available and known (e.g. image translations of Hamiltonian systems), the goodness-of-fit score R<sup>2</sup> can be used per dimension to show how well the latent system of the Neural SSM captures the dynamics in an interpretable way<sup>[1,3]</sup>. This is easiest to leverage in linear transition dynamics. 
<!-- While not studied in rigor, it's possible that non-linear dynamics may be more difficult to interpret with R<sup>2</sup> given the potential complexity of capturing fit quality between a high-dimensional non-linear latent state and single-dimensional physical variables. -->
Ref. [1], while containing linear transition dynamics, mentioned the possibility of non-linear regression via vanilla neural networks, though this may run into concerns of regressor capacity and data sizes. Additionally, incorporating metrics derived from latent disentanglement learning may provide stronger evaluation capabilities.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/171448933-45983b30-b2fd-4efc-b058-d8f78050ec53.png" alt="dvbf latent interpretability" /></p>
<p align='center'>Fig N. DVBF Latent Space Visualization for R<sup>2</sup> scores<sup>[1,3]</sup>.</p>


<!-- EXPERIMENTS -->
<a name="experiments"></a>
# Experiments

This section details some experiments that evaluate the fundamental aspects of Neural SSMs and the effects of the framework decisions one can take. Trained model checkpoints and hyperparameter files are provided for each experiment under ```experiments/model```. Evaluations are done with the metrics discussed above, as well as visualizations of animated trajectories over time and latent walk visualizations.

<!-- HYPERPARAMETER TUNING -->
<a name="hyperparameters"></a>
## Hyperparameter Tuning

<!-- HAMILTONIAN -->
<a name="hamiltonian"></a>
## Hamiltonian Systems (Pendulum + Mass-Spring)

<!-- Miscellaneous -->
<a name="misc"></a>
# Miscellaneous

This section just consists of to-do's within the repo, contribution guidelines, and a section on how to find the references used throughout the repo.

<!-- TO-DO -->
<a name="todo"></a>
## To-Do

<h4>Repository-wise</h4>

- Make a ```requirements.txt``` file for an Anaconda environment

<h4>Model-wise</h4>

- Implement a version of DKF under ```state_estimation/```
- Implement a version of VRNN under ```state_estimation/```
- Implement a version of KVAE under ```system_identification``` (or any linear-gaussian SSM method)
- Implement a version of DVBF under ```system_identification```

<h4>Evaluation-wise</h4>

- Implement latent walk visualizations against data-space observations (like in DVBF)

<h4>README-wise</h4>

- Add guidelines for an ```Experiment``` section highlighting experiments performed in validating the models
- Add a section explaining for ```Meta-Learning``` works in Neural SSMs

<!-- CONTRIBUTIONS -->
<a name="contributions"></a>
## Contributions
Contributions are welcome and encouraged! If you have an implementation of a latent dynamics function you think would be relevant and add to the conversation, feel free to submit an Issue or PR and we can discuss its incorporation. Similarly, if you feel an area of the README is lacking or contains errors, please put up a README editing PR with your suggested updates. Even tackling items on the To-Do would be massively helpful!

<!-- REFERENCES  -->
<a name="references"></a>
## References
1. Maximilian Karl, Maximilian Soelch, Justin Bayer, and Patrick van der Smagt. Deep variational bayes filters: Unsupervised learning of state space models from raw data. In International Conference on Learning Representations, 2017.
2. Marco Fraccaro, Simon Kamronn, Ulrich Paquetz, and OleWinthery. A disentangled recognition and nonlinear dynamics model for unsupervised learning. In Advances in Neural Information Processing Systems, 2017.
3. Alexej Klushyn, Richard Kurle, Maximilian Soelch, Botond Cseke, and Patrick van der Smagt. Latent matters: Learning deep state-space models. Advances in Neural Information Processing Systems, 34, 2021.
4. Aleksandar Botev, Andrew Jaegle, Peter Wirnsberger, Daniel Hennes, and Irina Higgins. Which priors matter? benchmarking models for learning latent dynamics. In Advances in Neural Information Processing Systems, 2021.
5. C. Yildiz, M. Heinonen, and H. Lahdesmaki. ODE2VAE: Deep generative second order odes with bayesian neural networks. In Neural Information Processing Systems, 2020.
6. Batuhan Koyuncu. Analysis of ode2vae with examples. arXiv preprint arXiv:2108.04899, 2021.
7. Rahul G. Krishnan, Uri Shalit, and David Sontag. Structured inference networks for nonlinear state space models. In Association for the Advancement of Artificial Intelligence, 2017.
8. Daehoon Gwak, Gyuhyeon Sim, Michael Poli, Stefano Massaroli, Jaegul Choo, and Edward Choi. Neural ordinary differential equations for intervention modeling. arXiv preprint arXiv:2010.08304, 2020.
9. Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. In Advances in Neural Information Processing Systems, 2015.
10. Yulia Rubanova, Ricky T. Q. Chen, and David Duvenaud. Latent odes for irregularly-sampled time series. In Neural Information Processing Systems, 2019.
11. Tsuyoshi Ishizone, Tomoyuki Higuchi, and Kazuyuki Nakamura. Ensemble kalman variational objectives: Nonlinear latent trajectory inference with a hybrid of variational inference and ensemble kalman filter. arXiv preprint arXiv:2010.08729, 2020.
12. Justin Bayer, Maximilian Soelch, Atanas Mirchev, Baris Kayalibay, and Patrick van der Smagt. Mind the gap when conditioning amortised inference in sequential latent-variable models. arXiv preprint arXiv:2101.07046, 2021.
13. Ðor ̄de Miladinovi ́c, Muhammad Waleed Gondal, Bernhard Schölkopf, Joachim M Buhmann, and Stefan Bauer. Disentangled state space representations. arXiv preprint arXiv:1906.03255, 2019.
14. Zeshan Hussain, Rahul G. Krishnan, and David Sontag. Neural pharmacodynamic state space modeling, 2021.
15. Francesco Paolo Casale, Adrian Dalca, Luca Saglietti, Jennifer Listgarten, and Nicolo Fusi.Gaussian process prior variational autoencoders. Advances in neural information processing systems, 31, 2018.
16. Yingzhen Li and Stephan Mandt. Disentangled sequential autoencoder. arXiv preprint arXiv:1803.02991, 2018.
17. Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular time series. Advances in Neural Information Processing Systems, 33:6696-6707, 2020.
18. Edward De Brouwer, Jaak Simm, Adam Arany, and Yves Moreau. Gru-ode-bayes: Continuous modeling of sporadically-observed time series. Advances in neural information processing systems, 32, 2019.
19. Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryull Sohn, Xunyu Lin, and Honglak Lee. Learning to generate long-term future via hierarchical prediction. In international conference on machine learning, pages 3560–3569. PMLR, 2017
20. Xiajun Jiang, Ryan Missel, Maryam Toloubidokhti, Zhiyuan Li, Omar Gharbia, John L Sapp, and Linwei Wang. Label-free physics-informed image sequence reconstruction with disentangled spatial-temporal modeling. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 361–371. Springer, 2021.
21. Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.
22. Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. Advances in neural information processing systems, 28, 2015.
23. Junbo Zhang, Yu Zheng, and Dekang Qi. Deep spatio-temporal residual networks for citywide crowd flows prediction. In Thirty-first AAAI conference on artificial intelligence, 2017.
24. Yong Han, Shukang Wang, Yibin Ren, Cheng Wang, Peng Gao, and Ge Chen. Predicting station-level short-term passenger flow in a citywide metro network using spatiotemporal graph convolutional neural networks. ISPRS International Journal of Geo-Information, 8(6):243, 2019