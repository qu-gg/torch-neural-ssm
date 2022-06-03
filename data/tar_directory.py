"""
@file tar_directory.py
@author Ryan Missel

Handles taking in a folder of individual .npy samples and combining them into N number of tar files, or "shards,
for WebDataset handling
"""
import os
import random
import tarfile
from tqdm import tqdm

# Get file list and then shuffle it
direct = ""
num_steps = 65
mode = "test"

file_list = os.listdir("{}/{}/".format(direct, mode))
random.shuffle(file_list)

n_shards = 1000
elements_per_shard = len(file_list) // n_shards

if not os.path.exists("{}/{}_tars/".format(direct, mode)):
    os.mkdir("{}/{}_tars/".format(direct, mode))

for n in tqdm(range(n_shards)):
    with tarfile.open(direct + "/{}_tars".format(mode) + "/{0:04}.tar".format(n), "w:gz") as tar:
        for file in file_list[n * elements_per_shard: (n + 1) * elements_per_shard]:
            tar.add("{}/{}/{}".format(direct, mode, file))
