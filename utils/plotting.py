"""
@file plotting.py

Holds general plotting functions for reconstructions of the bouncing ball dataset
"""
import numpy as np
import matplotlib.pyplot as plt


def show_images(images, preds, out_loc, num_out=None):
    """
    Constructs an image of multiple time-series reconstruction samples compared against its relevant ground truth
    Saves locally in the given out location
    :param images: ground truth images
    :param preds: predictions from a given model
    :out_loc: where to save the generated image
    :param num_out: how many images to stack. If None, stack all
    """
    assert len(images.shape) == 4       # Assert both matrices are [Batch, Timesteps, H, W]
    assert len(preds.shape) == 4
    assert type(num_out) is int or type(num_out) is None

    # Make sure objects are in numpy format
    if not isinstance(images, np.ndarray):
        images = images.cpu().numpy()
        preds = preds.cpu().numpy()

    # Splice to the given num_out
    if num_out is not None:
        images = images[:num_out]
        preds = preds[:num_out]

    # Iterate through each sample, stacking into one image
    out_image = None
    for idx, (gt, pred) in enumerate(zip(images, preds)):
        # Pad between individual timesteps
        gt = np.pad(gt, pad_width=(
            (0, 0), (5, 5), (0, 1)
        ), constant_values=1)

        gt = np.hstack([i for i in gt])

        # Pad between individual timesteps
        pred = np.pad(pred, pad_width=(
            (0, 0), (0, 10), (0, 1)
        ), constant_values=1)

        # Stack timesteps into one image
        pred = np.hstack([i for i in pred])

        # Stack gt/pred into one image
        final = np.vstack((gt, pred))

        # Stack into out_image
        if out_image is None:
            out_image = final
        else:
            out_image = np.vstack((out_image, final))

    # Save to out location
    plt.imsave(out_loc, out_image, cmap='gray')


def get_embedding_trajectories(embeddings, states, out_loc):
    """
    Handles getting trajectory plots of the embedded states against the true physical states
    :param embeddings: vector states over time
    :param states: ground truth physical parameter states
    """
    # Make sure objects are in numpy format
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.cpu().numpy()
        states = states.cpu().numpy()

    # Get embedding trajectory plots
    for idx, embedding in enumerate(np.swapaxes(embeddings, 1, 0)):
        plt.plot(embedding, label=f"Dim {idx}")

    plt.title("Vector State Trajectories")
    plt.savefig(f"{out_loc}/trajectories_embeddings.png")
    plt.close()

    # Get physical state trajectories
    for idx, embedding in enumerate(np.swapaxes(states, 1, 0)):
        plt.plot(embedding, label=f"Dim {idx}")

    plt.title("GT State Trajectories")
    plt.savefig(f"{out_loc}/trajectories_gt.png")
    plt.close()
