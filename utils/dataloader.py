"""
@file dataloader.py
@author Ryan Missel

Holds the WebDataset classes for the available datasets
"""
import os
import webdataset as wds
import pytorch_lightning


class Dataset(pytorch_lightning.LightningDataModule):
    def __init__(self, args, batch_size=32, workers=0):
        super(Dataset, self).__init__()
        # Get the number of shards used for the given data size
        num_shards = int(args.dataset_percent * 999)

        # Build shard paths for the training data
        bucket = os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}/train_tars/"
        shards = "{000.." + str(num_shards) + "}.tar"
        self.training_urls = os.path.join(bucket, shards)

        # Build shard paths for the validation data
        bucket = os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}/val_tars/"
        shards = "{000.." + str(num_shards) + "}.tar"
        self.validation_urls = os.path.join(bucket, shards)

        # Build shard paths for the testing data
        bucket = os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}/test_tars/"
        shards = "{000.." + str(num_shards) + "}.tar"
        self.testing_urls = os.path.join(bucket, shards)

        # Various parameters
        self.batch_size = batch_size
        self.num_workers = workers

    def make_loader(self, urls, mode="train"):
        """
        Builds and shuffles the WebDataset and constructs the WebLoader
        :param urls: local file paths of the shards
        :param mode: whether to shuffle the WDS or not
        :return: WebLoader object
        """
        shuffle = 1000 if mode == "train" else 0

        if mode == "train":
            dataset = (
                wds.WebDataset(urls, shardshuffle=True)
                    .shuffle(shuffle)
                    .decode("rgb")
                    .to_tuple("npz")
                    .batched(self.batch_size, partial=False)
            )
        else:
            dataset = (
                wds.WebDataset(urls, shardshuffle=False)
                    .decode("rgb")
                    .to_tuple("npz")
                    .batched(self.batch_size, partial=False)
            )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers
        )

        if mode == "train":
            loader.unbatched().shuffle(1000).batched(self.batch_size, partial=False)
        return loader

    def train_dataloader(self):
        """ Getter function that builds and returns the training dataloader """
        print(f"=> Loading training urls: {self.training_urls}")
        return self.make_loader(self.training_urls, "train")

    def val_dataloader(self):
        """ Getter function that builds and returns the validation dataloader """
        print(f"=> Loading validation urls: {self.validation_urls}")
        return self.make_loader(self.validation_urls, "val")

    def test_dataloader(self):
        """ Getter function that builds and returns the testing dataloader """
        print(f"=> Loading testing urls: {self.testing_urls}")
        return self.make_loader(self.testing_urls, "test")
