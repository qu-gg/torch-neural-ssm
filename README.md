<h2 align='center'>torchssm</h2>
<h3 align='center'>Neural State-Space Models and Latent Dynamic Functions <br> for High-Dimensional Generative Time-series Modelling</h3>

<a name="about"/>

## About this Repository
This repository is meant to conceptually introduce and highlight implementation considerations for the recent class of models called <b>Neural State-Space Models (Neural SSMs)</b>. They leverage the classical state-space models with the flexibility of deep learning to approach high-dimensional generative time-series modelling and learning latent dynamics functions.

Included is an abstract PyTorch-Lightning training class structure with specific latent dynamic functions that inherit it, as well as common metrics used in their evaluation and training examples on common datasets. Further broken down via implementation is the distinction between <i>system identification</i> and <i>state estimation</i> approaches, which are reminiscent of their classic SSM counterparts and arise from fundamental differences in the underlying choice of probailistic graphical model (PGM).

![SavingFile](https://user-images.githubusercontent.com/32918812/169753112-bc849b24-fe13-4975-8697-fea95bb19fb5.png)
<p align='center'>Fig 1. Schematic of the two PGM forms of Neural SSMs.</p>

<!-- CITATION -->
<a name="citation"/>

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

<a name="toc"/>

## Table of Contents
- [About](#about)
- [Citation](#citation)
- [Table of Contents](#toc)
- [Background](#background)
  - [What are Neural SSMs?](#neuralSSMwhat)
  - [Choice of SSM PGM](#pgmChoice)
  - [System Controls, u<sub>t</sub>](#ssmControls)
- [Implementation](#implementation)
  - [Data](#data)
  - [Models](#models)
  - [Metrics](#metrics)
- [Miscellaneous](#misc)
  - [To-Do](#todo)
  - [Contributions](#contributions)
  - [References](#references)

<!-- BACKGROUND -->
<a name="background"/>

# Background

This section gives an introduction to the concept of Neural SSMs, some common considerations and limitations, and active areas of research. This section assumes some familiarity with state-space models, though not much is required to gain a conceptual understanding if one is already coming from a latent modelling perspective or Bayesian learning. Resources are a plenty out there considering the width and depth of state-space usage, however this <a href="https://www.youtube.com/watch?v=hpeKrMG-WP0">resource</a> is a good starting point.

<!-- Neural SSM INTRO -->
<a name="neuralSSMwhat"/>

## What are Neural SSMs?
An extension of classical state-space models, they - at their core - consist of a dynamic function of some latent states <b>z_k</b> and their emission to observations <b>x_k</b>, realized through the equations:
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169743189-057f52a5-8a08-4616-9516-3c60aca86b28.png" alt="neural ssm equations" /></p>
where <b>θ_z</b> represents the parameters of the latent dynamic function. The precise form of these functions can vary significantly - from deterministic or stochastic, linear or non-linear, and discrete or continuous.
<p> </p>
Due to their explicit differentiation of transition and emission and leveraging of structured equations, they have found success in learning interpretable latent dynamic spaces<sup>[1,2,3]</sup>, identifying physical systems from non-direct features<sup>[4,5,6]</sup> and uses in counterfactual forecasting<sup>[7,8,14]</sup>. 
<p> </p>
Given the fast pace of progress in latent dynamics modelling over recent years, many models have been presented under a variety of terminologies and proposed frameworks - examples being variational latent recurrent models<sup>[5,9,10,11,12]</sup>, deep state space models<sup>[1,2,3,7,13,14]</sup>, and deterministic encoding-decoding models<sup>[4,15,16]</sup>. Despite differences in appearance, they all adhere to the same conceptual framework of latent variable modelling and state-space disentanglement. As such, here we unify them under the term of Neural SSMs and segment them into the two base choices of probabilistic graphical models that they adhere to: <i>system identification</i> and <i>state estimation</i>. We highlight each PGM's properties and limitations with experimental evaluations on benchmark datasets.

<!-- PGM CHOICES -->
<a name="pgmChoice"/>

## Choice of PGM - System Identification vs State Estimation
The PGM associated with each approach is determined by the latent variable chosen for inference.

<b>System states as latent variables</b>: The intuitive choice for the latent variable is the latent state <b>z_k</b> that underlies <b>x_k</b>, given that it is already latent in the system and is directly associated with the observations. The PGM of this form is shown under Fig. 1A

<!-- CONTROLS -->
<a name="ssmControls"/>

## Use of System Controls, u<sub>t</sub>

Insofar we have ignored another common and important component of state-space modelling, the incorporation of external controls <i>u</i> that affect the transition function of the state. Controls represent factors that influence the trajectory of a system but are not direct features of the object/system being modelled. For example, an external force such as friction acting on a moving ball or medications given to a patient could be considered controls<sup>[8,14]</sup>. These allow an additional layer of interpretability in SSMs and even enable counterfactual reasoning; i.e. given the current state, what does its trajectory look like under varying control inputs going forwards? This has myriad uses in medical modelling with counterfactual medicine<sup>[14]</sup> or physical system simulations<sup>[8]</sup>
<p> </p>

For Neural SSMs, a variety of approaches have been taken thus far dependent on the type of latent transition function utilized. 
<p> </p>

<b>Linear Dynamics</b>: In latent dynamics still parameterized by traditional linear gaussian transition functions, control incorporation is as easy as the addition of another transition matrix <b>B<sub>t</sub></b> that modifies a control input <b>u<sub>t</sub></b> at each timestep<sup>[1,2,4,7]</sup>. 

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170075684-2ba31e45-b66f-4d3c-aed6-9ab28def95d6.png" alt="linear control" /></p>
<p align='center'>Fig N. Example of control input in a linear transition function<sup>[1]<sup>.</p>
<p> </p>

<b>Non-Linear Dynamics</b>: In discrete non-linear transition matrices using either multi-layer perceptrons or recurrent cells, these can be leveraged by either concatenating it to the input vector before the network forward pass or as a data transformation in the form of element-wise addition and a weighted combination<sup>[10]</sup>. 

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170173582-a8158240-62d0-4b7e-8793-d1c796bc4a6c.png" alt="non-linear control" /></p>
<p align='center'>Fig N. Example of control input in a non-linear transition function<sup>[1]<sup>.</p>
<p> </p>

<b>Continuous Dynamics</b>: For incorporation into continuous latent dynamics functions, finding the best approaches is an ongoing topic of interest. Thus far, the reigning approaches are:

1. Directly jumping the vector field state with recurrent cells<sup>[18]</sup>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170078493-b7d10d50-d252-4258-bed7-f7c2ae1080b9.png" alt="jump control" /></p>
    
2. Influencing the vector field gradient (e.g. neural controlled differential equations)<sup>[17]</sup>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170079172-b2dd6376-628d-4e15-8282-4ee296cd5b89.png" alt="gradient control" /></p>
    
3. Introducing another dynamics mechanism, continuous or otherwise (e.g. neural ODE or attention blocks), that are combined with the latent trajectory <b>z<sub>1:T</sub></b> into an auxiliary state <b>h<sub>1:T</sub></b><sup>[8,14]</sup>.
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/170077468-f183e75f-3ad0-450e-b402-6718087c9b9c.png" alt="continuous control" /></p>

    
<!-- IMPLEMENTATION -->
<a name="implementation"/>

# Implementation
    
In this section, details on model implementation and the datasets/metrics used are detailed. The models and datasets used throughout this repo are solely grayscale physics datasets with underlying Hamiltonian laws, such as pendulum and mass spring sets. Extensions to color images and non-pixel based tasks (or even graph-based data!) is easily done in this framework, as the only architecture change need is the structure of the encoder and decoder networks as the state propagation happens solely in a latent space.
    
    
<!-- DATA -->
<a name="data"/>

## Data

<!-- MODELS -->
<a name="models"/>

## Models

<!-- METRICS -->
<a name="metrics"/>

## Metrics

<b> Mean Squared Error (MSE)</b>: A common metric used in video and image tasks where its use is in per-frame average over individual pixel error. While a multitude of papers solely use plots of frame MSE over time as an evaluation metric, it is insufficient for comparison between models - especially in cases where the dataset contains a small object for reconstruction<sup>[4]</sup>. This is especially prominent in tasks of system identification where a model that fails to predict long-term may end up with a lower average MSE than a model that has better generation but is slightly off in its object placement. 

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169945139-ebcfc6e9-14d5-4a88-bc38-a9ed41ff5dfc.png" alt="mse equation" /></p>
<p align='center'>Fig 2. Per-Frame MSE Equation.</p>

<b>Valid Prediction Time (VPT)</b>: Introduced in [4], the VPT metric is an advance on latent dynamics evaluation over pure pixel-based MSE metrics. For each prediction sequence, the per-pixel MSE is taken over the frames individually and the minimum timestep in which the MSE surpasses a pre-defined epsilon is considered the 'valid prediction time.' The resulting mean number over the samples is often normalized over the total prediction timesteps to get a percentage of valid predictions.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169942657-5208afb5-6faf-4d47-b9a2-ef0a89a5fc9f.png" alt="vpt equation" /></p>
<p align='center'>Fig 3. Per-Sequence VPT Equation.</p>

<b>Object Distance (DST)</b>: Another potential metric to support evaluation (useful in image-based physics forecasting tasks) is using the Euclidean distance between the estimated center of the predicted object and its ground truth center. Otsu's Thresholding <a href="https://en.wikipedia.org/wiki/Otsu%27s_method">method</a> can be applied to grayscale output images to get binary predictions of each pixel and then the average pixel location of all the "active" pixels can be calculated. This approach can help alleviate the prior MSE issues of metric imbalance as the maximum Euclidean error of a given image space can be applied to model predictions that fail to have any pixels over Otsu's threshold.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169954436-74d02fdc-0ab3-4d2e-b2b4-b595e35144a0.png" alt="dst equation" /></p>
<p align='center'>Fig 4. Per-Frame DST Equation.</p>
where R<sup>N</sup> is the dimension of the output (e.g. number of image channels) and s, s<sub>hat</sub> are the subsets of "active" predicted pixels.
<p> </p>
<b>Valid Prediction Distance (VPD)</b>: Similar in spirit to how VPT leverages MSE, VPD is the minimum timestep in which the DST metric surpasses a pre-defined epsilon. This is useful in tracking how long a model can generate an object in a physical system before either incorrect trajectories and/or error accumulation cause significant divergence.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169961714-2d007dbd-92f2-4ec7-aff0-3c383f21e919.png" alt="vpd equation" /></p>
<p align='center'>Fig 5. Per-Sequence VPD Equation.</p>

<!-- Miscellaneous -->
<a name="misc"/>

# Miscellaneous

This section just consists of to-do's within the repo, contribution guidelines, and a section on how to find the references used throughout the repo.

<!-- TO-DO -->
<a name="todo"/>

## To-Do
<h4>Repository-wise</h4>

- Make a ```state_estimation/``` folder in ```models/```
- Make a ```system_identification``` folder in ```models``` and shift current models to it

<h4>Model-wise</h4>

- Implement a version of DKF under ```state_estimation/```
- Implement a version of VRNN under ```state_estimation/```
- Implement a version of KVAE under ```system_identification``` (or any linear-gaussian ssm method)
- Implement a version of DVBF under ```system_identification```

<h4>Evaluation-wise</h4>

- Implement R^2 coefficient statistics from latent state to physical variables
- Implement latent walk visualizations against data-space observations (like in DVBF)

<h4>README-wise</h4>

- Complete ```Introduction``` section with PGM explanations + examples
- Complete ```Implementation``` section
    - Data section: datasets, dataloaders, data generators, common datasets
    - Model section: description of abstract class, PyTorch-Lightning training, dynamics class inheritance, etc
- Add guidelines for an ```Experiment``` section highlighting experiments performed in validating the models

<!-- CONTRIBUTIONS -->
<a name="contributions"/>

## Contributions
Contributions are welcome and encouraged! If you have an implementation of a latent dynamics function you think would be relevant and add to the conversation, feel free to submit an Issue or PR and we can discuss its incorporation. Similarly, if you feel an area of the README is lacking or contains errors, please put up a README editing PR with your suggested updates. Even tackling items on the To-Do would be massively helpful!

<!-- REFERENCES  -->
<a name="references"/>

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
