<h2 align='center'>torchssm</h2>
<h3 align='center'>Neural State-Space Models and Latent Dynamic Functions <br> for High-Dimensional Generative Time-series Modelling</h3>

## About
This repository is meant to conceptually introduce and highlight implementation considerations for the recent class of models called <b>Neural State-Space Models (Neural SSMs)</b>. They leverage the classical state-space models with the flexibility of deep learning to approach high-dimensional generative time-series modelling and learning latent dynamics functions.

Included is an abstract PyTorch-Lightning training class structure with specific latent dynamic functions that inherit it, as well as common metrics used in their evaluation and training examples on common datasets. Further broken down via implementation is the distinction between <i>system identification</i> and <i>state estimation</i> approaches, which are reminiscent of their classic SSM counterparts and arise from fundamental differences in the underlying choice of probailistic graphical model.

![Fig1](https://user-images.githubusercontent.com/32918812/169742726-2f1f849f-1ec2-4815-ae24-411b842c71b7.png)


## What are Neural SSMs?
An extension of classical state-space models, they at their core consist of a dynamic function of some latent states <b>z_k</b> and their emission to observations <b>x_k</b>, realized through the equations:
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169743189-057f52a5-8a08-4616-9516-3c60aca86b28.png" alt="centered image" /></p>
where <b>θ_z</b> represents the parameters of the latent dynamic function. The precise form of these functions can vary significantly - from deterministic or stochastic, linear or non-linear, and discrete or continuous.

Due to their explicit differentiation of transition and emission and leveraging of structured equations, they have found success in learning interpretable latent dynamic spaces \[DVBF, KVAE, EKVAE\], identifying physical systems from non-direct features \[PriorsMatter, ODE2VAE, ODE2VAE Exp\] and uses in counterfactual forecasting \[DKF, IMODE\]. 

Given the past pace of progress in latent dynamics modelling over recent years, many models have been presented under a variety of terminologies and proposed frameworks - examples being variational latent recurrent models \[VRNN, Latent ODE, ODE2VAE, EnKO, MindTheGap\], deep state space models, and deterministic encoding-decoding models \[PriorMatters, GPPVAE, DisSeqAE\]. Despite differences in appearance, they all adhere to the same conceptual framework of latent variable modelling and state-space disentanglement. As such, here we unify them under the term of Neural SSMs and segment them into the two base choices of probabilistic graphical models that they adhere to: <i>system identification</i> and <i>state estimation</i>. We highlight each PGM's properties and limitations with experimental evaluations on benchmark datasets.

## Choice of PGM - System Identification vs State Estimation
