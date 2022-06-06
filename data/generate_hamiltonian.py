"""
@file generate_date.py
@url https://github.com/deepmind/dm_hamiltonian_dynamics_suite/tree/master/dm_hamiltonian_dynamics_suite

Adapted from the DeepMind Hamiltonian Dynamics Suite for usage with WebDataset
Requires Linux and JAX to run, as well as a local folder structure of dm_hamiltonian_dynamics_suite.
As such, it won't run straight in this repository, however is here for demonstration purposes.
"""
from subprocess import getstatusoutput
from matplotlib import pyplot as plt
from matplotlib import animation as plt_animation
import numpy as np
from jax import config as jax_config

import os
import random
import tarfile
from tqdm import tqdm

jax_config.update("jax_enable_x64", True)

from dm_hamiltonian_dynamics_suite import load_datasets
from dm_hamiltonian_dynamics_suite import datasets

# @title Helper functions
DATASETS_URL = "gs://dm-hamiltonian-dynamics-suite"
DATASETS_FOLDER = "./datasets"  # @param {type: "string"}
os.makedirs(DATASETS_FOLDER, exist_ok=True)


def download_file(file_url, destination_file):
    print("Downloading", file_url, "to", destination_file)
    command = f"gsutil cp {file_url} {destination_file}"
    status_code, output = getstatusoutput(command)
    if status_code != 0:
        raise ValueError(output)


def download_dataset(dataset_name: str):
    """Downloads the provided dataset from the DM Hamiltonian Dataset Suite"""
    destination_folder = os.path.join(DATASETS_FOLDER, dataset_name)
    dataset_url = os.path.join(DATASETS_URL, dataset_name)
    os.makedirs(destination_folder, exist_ok=True)
    if "long_trajectory" in dataset_name:
        files = ("features.txt", "test.tfrecord")
    else:
        files = ("features.txt", "train.tfrecord", "test.tfrecord")
    for file_name in files:
        file_url = os.path.join(dataset_url, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        if os.path.exists(destination_file):
            print("File", file_url, "already present.")
            continue
        download_file(file_url, destination_file)


def unstack(value: np.ndarray, axis: int = 0):
    """Unstacks an array along an axis into a list"""
    split = np.split(value, value.shape[axis], axis=axis)
    return [np.squeeze(v, axis=axis) for v in split]


def make_batch_grid(
        batch: np.ndarray,
        grid_height: int,
        grid_width: int,
        with_padding: bool = True):
    """Makes a single grid image from a batch of multiple images."""
    assert batch.ndim == 5
    assert grid_height * grid_width >= batch.shape[0]
    batch = batch[:grid_height * grid_width]
    batch = batch.reshape((grid_height, grid_width) + batch.shape[1:])
    if with_padding:
        batch = np.pad(batch, pad_width=[[0, 0], [0, 0], [0, 0],
                                         [1, 0], [1, 0], [0, 0]],
                       mode="constant", constant_values=1.0)
    batch = np.concatenate(unstack(batch), axis=-3)
    batch = np.concatenate(unstack(batch), axis=-2)
    if with_padding:
        batch = batch[:, 1:, 1:]
    return batch


def plot_animattion_from_batch(
        batch: np.ndarray,
        grid_height,
        grid_width,
        with_padding=True,
        figsize=None):
    """Plots an animation of the batch of sequences."""
    if figsize is None:
        figsize = (grid_width, grid_height)
    batch = make_batch_grid(batch, grid_height, grid_width, with_padding)
    batch = batch[:, ::-1]
    fig = plt.figure(figsize=figsize)
    plt.close()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    img = ax.imshow(batch[0])

    def frame_update(i):
        i = int(np.floor(i).astype("int64"))
        img.set_data(batch[i])
        return [img]

    anim = plt_animation.FuncAnimation(
        fig=fig,
        func=frame_update,
        frames=np.linspace(0.0, len(batch), len(batch) * 5 + 1)[:-1],
        save_count=len(batch),
        interval=10,
        blit=True
    )
    return anim


def plot_sequence_from_batch(
        batch: np.ndarray,
        t_start: int = 0,
        with_padding: bool = True,
        fontsize: int = 20):
    """Plots all of the sequences in the batch."""
    n, t, dx, dy = batch.shape[:-1]
    xticks = np.linspace(dx // 2, t * (dx + 1) - 1 - dx // 2, t)
    xtick_labels = np.arange(t) + t_start
    yticks = np.linspace(dy // 2, n * (dy + 1) - 1 - dy // 2, n)
    ytick_labels = np.arange(n)
    batch = batch.reshape((n * t, 1) + batch.shape[2:])
    batch = make_batch_grid(batch, n, t, with_padding)[0]
    plt.imshow(batch.squeeze())
    plt.xticks(ticks=xticks, labels=xtick_labels, fontsize=fontsize)
    plt.yticks(ticks=yticks, labels=ytick_labels, fontsize=fontsize)


def visalize_dataset(
        dataset_path: str,
        sequence_lengths: int = 60,
        grid_height: int = 2,
        grid_width: int = 5):
    """Visualizes a dataset loaded from the path provided."""
    split = "test"
    batch_size = grid_height * grid_width
    dataset = load_datasets.load_dataset(
        path=dataset_path,
        tfrecord_prefix=split,
        sub_sample_length=sequence_lengths,
        per_device_batch_size=batch_size,
        num_epochs=None,
        drop_remainder=True,
        shuffle=False,
        shuffle_buffer=100
    )
    sample = next(iter(dataset))
    batch_x = sample['x'].numpy()
    batch_image = sample['image'].numpy()
    # Plot real system dimensions
    plt.figure(figsize=(24, 8))
    for i in range(batch_x.shape[-1]):
        plt.subplot(1, batch_x.shape[-1], i + 1)
        plt.title(f"Samples from dimension {i + 1}")
        plt.plot(batch_x[:, :, i].T)
    plt.show()
    # Plot a sequence of 50 images
    plt.figure(figsize=(30, 10))
    plt.title("Samples from 50 steps sub sequences.")
    plot_sequence_from_batch(batch_image[:, :50])
    plt.show()
    # Plot animation
    return plot_animattion_from_batch(batch_image, grid_height, grid_width)


# Generate dataset
print("Generating datasets....")
folder_to_store = "./generated_datasets"
dataset = "pendulum"
class_id = np.array([1])
dt = 0.1
num_steps = 1000
steps_per_dt = 1
num_train = 1000
num_test = 2000
overwrite = True
datasets.generate_full_dataset(
    folder=folder_to_store,
    dataset=dataset,
    dt=dt,
    num_steps=num_steps,
    steps_per_dt=steps_per_dt,
    num_train=num_train,
    num_test=num_test,
    overwrite=overwrite,
)
dataset_full_name = dataset + "_dt_" + str(dt).replace(".", "_")
dataset_output_path = dataset + "_{}samples_{}steps".format(num_train, num_steps) + "_dt" + str(dt).replace(".", "")
dataset_path = os.path.join(folder_to_store, dataset_full_name)
visalize_dataset(dataset_path)

if not os.path.exists("data_out/{}".format(dataset_output_path)):
    os.mkdir("data_out/{}".format(dataset_output_path))

"""
Training Generation
"""
print("Converting training files...")
if not os.path.exists("data_out/{}/train/".format(dataset_output_path)):
    os.mkdir("data_out/{}/train/".format(dataset_output_path))

loaded_dataset = load_datasets.load_dataset(
    path=dataset_path,
    tfrecord_prefix="train",
    sub_sample_length=num_steps,
    per_device_batch_size=1,
    num_epochs=1,
    drop_remainder=True,
    shuffle=True,
    shuffle_buffer=100
)

images = None
xs = None
for idx, sample in enumerate(loaded_dataset):
    image = sample['image'][0].numpy()

    # (32, 32, 3) -> (3, 32, 32)
    image = np.swapaxes(image, 2, 3)
    image = np.swapaxes(image, 1, 2)

    # Just grab the R channel
    image = image[:, 0, :, :]

    x = sample['x'].numpy()

    np.savez("data_out/{}/train/{}.npz".format(dataset_output_path, idx), image=image, x=x, class_id=class_id)

# Get file list and then shuffle it
file_list = os.listdir("data_out/" + dataset_output_path + "/train/")
random.shuffle(file_list)

if not os.path.exists("data_out/" + dataset_output_path + '/train_tars/'):
    os.mkdir("data_out/" + dataset_output_path + '/train_tars/')

n_shards = 200
elements_per_shard = len(file_list) // n_shards

for n in tqdm(range(n_shards)):
    with tarfile.open("data_out/" + dataset_output_path + "/train_tars/train{0:03}.tar".format(n), "w:gz") as tar:
        for file in file_list[n * elements_per_shard: (n + 1) * elements_per_shard]:
            tar.add("data_out/" + dataset_output_path + "/train/{}".format(file))

""" 
Testing Generation 
"""
print("Converting testing files...")
if not os.path.exists("data_out/{}/test/".format(dataset_output_path)):
    os.mkdir("data_out/{}/test/".format(dataset_output_path))
loaded_dataset = load_datasets.load_dataset(
    path=dataset_path,
    tfrecord_prefix="test",
    sub_sample_length=num_steps,
    per_device_batch_size=1,
    num_epochs=1,
    drop_remainder=True,
    shuffle=True,
    shuffle_buffer=100
)

images = None
xs = None
for idx, sample in enumerate(loaded_dataset):
    image = sample['image'][0].numpy()

    # (32, 32, 3) -> (3, 32, 32)
    image = np.swapaxes(image, 2, 3)
    image = np.swapaxes(image, 1, 2)

    # Just grab the R channel
    image = image[:, 0, :, :]

    x = sample['x'].numpy()
    np.savez("data_out/{}/test/{}.npz".format(dataset_output_path, idx), image=image, x=x, class_id=class_id)

# Get file list and then shuffle it
file_list = os.listdir("data_out/" + dataset_output_path + "/test/")
random.shuffle(file_list)

if not os.path.exists("data_out/" + dataset_output_path + '/test_tars/'):
    os.mkdir("data_out/" + dataset_output_path + '/test_tars/')

n_shards = 200
elements_per_shard = len(file_list) // n_shards

for n in tqdm(range(n_shards)):
    with tarfile.open("data_out/" + dataset_output_path + "/test_tars/test{0:03}.tar".format(n), "w:gz") as tar:
        for file in file_list[n * elements_per_shard: (n + 1) * elements_per_shard]:
            tar.add("data_out/" + dataset_output_path + "/test/{}".format(file))
