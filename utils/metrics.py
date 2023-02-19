"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models, including
Valid Prediction Time (VPT), Valid Prediction Distance (VPD), etc.
"""
import numpy as np
import torch
import torch.nn as nn

from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def vpt(gt, preds, epsilon=0.010, **kwargs):
    """
    Computes the Valid Prediction Time metric, as proposed in https://openreview.net/pdf?id=qBl8hnwR0px
    VPT = argmin_t [MSE(gt, pred) > epsilon]
    :param gt: ground truth sequences
    :param preds: model predicted sequences
    :param epsilon: threshold for valid prediction
    """
    # Ensure on CPU and numpy
    if not isinstance(gt, np.ndarray):
        gt = gt.cpu().numpy()
        preds = preds.cpu().numpy()

    # Get dimensions
    _, timesteps, height, width = gt.shape

    # Get pixel_level MSE at each timestep
    mse = (gt - preds) ** 2
    mse = np.sum(mse, axis=(2, 3)) / (height * width)

    # Get VPT
    vpts = []
    for m in mse:
        # Get all indices below the given epsilon
        indices = np.where(m < epsilon)[0] + 1

        # If there are none below, then add 0
        if len(indices) == 0:
            vpts.append(0)
            continue

        # Append last in list
        vpts.append(indices[-1])

    # Return VPT mean over the total timesteps
    return np.mean(vpts) / timesteps, np.std(vpts) / timesteps


def thresholding(preds, gt):
    """
    Thresholding function that converts gt and preds into binary images
    Activated prediction pixels are found via Otsu's thresholding function
    """
    N, T = gt.shape[0], gt.shape[1]
    res = np.zeros_like(preds)

    # For each sample and timestep, get Otsu's threshold and binarize gt and pred
    for n in range(N):
        for t in range(T):
            img = preds[n, t]
            otsu_th = np.max([0.32, threshold_otsu(img)])
            res[n, t] = (img > otsu_th).astype(np.float32)
            gt[n, t] = (gt[n, t] > 0.55).astype(np.float32)
    return res, gt


def dst(gt, preds, **kwargs):
    """
    Computes a Euclidean distance metric between the center of the ball in ground truth and prediction
    Activated pixels in the predicted are computed via Otsu's thresholding function
    :param gt: ground truth sequences
    :param preds: model predicted sequences
    """
    # Ensure on CPU and numpy
    if not isinstance(gt, np.ndarray):
        gt = gt.cpu().numpy()
        preds = preds.cpu().numpy()

    # Get shapes
    num_samples, timesteps, height, width = gt.shape

    # Apply Otsu thresholding function on output
    preds, gt = thresholding(preds, gt)

    # Loop over each sample and timestep to get the distance metric
    results = np.zeros([num_samples, timesteps])
    for n in range(num_samples):
        for t in range(timesteps):
            # Get all active predicted pixels
            a = preds[n, t]
            b = gt[n, t]
            pos_a = np.where(a == 1)
            pos_b = np.where(b == 1)

            # If there are in gt, add 0
            if pos_b[0].shape[0] == 0:
                results[n, t] = 0
                continue

            # Get gt center
            center_b = np.array([pos_b[0].mean(), pos_b[1].mean()])

            # Get center of predictions
            if pos_a[0].shape[0] != 0:
                center_a = np.array([pos_a[0].mean(), pos_a[1].mean()])
            # If no pixels above threshold, add the highest possible error in image space
            else:
                results[n, t] = np.sqrt(np.sum(np.array([height, width]) ** 2))
                continue

            # Get distance metric
            dist = np.sum((center_a - center_b) ** 2)
            dist = np.sqrt(dist)

            # Add to result
            results[n, t] = dist

    return np.mean(results), np.std(results), results


def vpd(output, target, **kwargs):
    """
    Computes the Valid Prediction Time metric, as proposed in https://openreview.net/forum?id=7C9aRX2nBf2
    VPD = argmin_t [DST(gt, pred) > epsilon]
    :param gt: ground truth sequences
    :param preds: model predicted sequences
    :param epsilon: threshold for valid prediction
    """
    epsilon = 10
    _, _, dsts = dst(output, target)
    B, T = dsts.shape
    vpdist = np.zeros(B)
    for i in range(B):
        idx = np.where(dsts[i, :] >= epsilon)[0]
        if idx.shape[0] > 0:
            vpdist[i] = np.min(idx)
        else:
            vpdist[i] = T

    # Return VPT mean over the total timesteps
    return np.mean(vpdist) / T, np.std(vpdist) / T


def reconstruction_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output[:, :kwargs['args'].generation_len] - target[:, :kwargs['args'].generation_len]) ** 2
    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2, 3))
    return np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)


def extrapolation_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for a number of steps past the length used in training """
    full_pixel_mses = (output[:, kwargs['args'].generation_len:] - target[:, kwargs['args'].generation_len:]) ** 2
    if full_pixel_mses.shape[1] == 0:
        return 0.0, 0.0

    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2, 3))
    return np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)


def r2fit(latents, gt_state, mlp=False):
    """
    Computes an R^2 fit value for each ground truth physical state dimension given the latent states at each timestep.
    Gets an average per timestep.
    :param latents: latent states at each timestep [BatchSize, TimeSteps, LatentSize]
    :param gt_state: ground truth physical parameters [BatchSize, TimeSteps, StateSize]
    :param mlp: whether to use a non-linear MLP regressor instead of linear regression
    """
    r2s = []

    # Ensure on CPU and numpy
    if not isinstance(latents, np.ndarray):
        latents = latents.cpu().numpy()
        gt_state = gt_state.cpu().numpy()

    # Convert to one large set of latent states
    latents = latents.reshape([latents.shape[0] * latents.shape[1], -1])
    gt_state = gt_state.reshape([gt_state.shape[0] * gt_state.shape[1], -1])

    # For each dimension of gt_state, get the R^2 value
    for sidx in range(gt_state.shape[-1]):
        gts = gt_state[:, sidx]

        # Whether to use LinearRegression or an MLP
        if mlp:
            reg = MLPRegressor().fit(latents, gts)
        else:
            reg = LinearRegression().fit(latents, gts)

        r2s.append(reg.score(latents, gts))

    # Return r2s for logging
    return r2s


def normalized_pixel_mse(gt, preds):
    """
    Handles getting the pixel MSE of a trajectory, but normalizes over the average intensity of the ground truth.
    This helps to be able to compare pixel MSE effectively over different domains rather than looking at pure intensity.
    :param gt: ground truth sequence [BS, TS, Dim1, Dim2]
    :param preds: predictions of the model [BS, TS, Dim1, Dim2]

    TODO: Make sure this metric calculation matches WhichPriorsMatter?
    """
    mse = nn.MSELoss(reduction='none')(gt, preds)
    mse = mse / torch.mean(gt ** 2)
    return mse.detach().cpu().numpy(), mse.mean([1, 2, 3]).mean().detach().cpu().numpy()
