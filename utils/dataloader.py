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
        shard_size = (args.dataset_size // 50) - 1

        # Build shard paths for the training data
        bucket = "C:/Users/rxm72/PycharmProjects/torchssm/data/{}/{}/train_tars/".format(args.dataset, args.dataset_ver)
        shards = "{000.." + str(shard_size) + "}.tar"
        self.training_urls = os.path.join(bucket, shards)
        print(self.training_urls)

        # Build shard paths for the validation data
        bucket = "C:/Users/rxm72/PycharmProjects/torchssm/data/{}/{}/test_tars/".format(args.dataset, args.dataset_ver)
        shards = "{000.." + str(shard_size) + "}.tar"
        self.validation_urls = os.path.join(bucket, shards)

        # Various parameters
        self.length = args.dataset_size // batch_size
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

        dataset = (
            wds.WebDataset(urls, shardshuffle=True)
            .shuffle(shuffle)
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

        loader.length = self.length
        loader.unbatched().shuffle(1000).batched(self.batch_size)
        return loader

    def train_dataloader(self):
        """ Getter function that builds and returns the training dataloader """
        return self.make_loader(self.training_urls, "train")

    def val_dataloader(self):
        """ Getter function that builds and returns the validation dataloader """
        return self.make_loader(self.validation_urls, "val")
