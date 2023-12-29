import torch.utils.data.distributed as dt

from lambdaLearn.Base.BaseSampler import BaseSampler


class DistributedSampler(BaseSampler):
    def __init__(self, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        """
        :param num_replicas: The Number of processes participating in distributed training.
        :param rank: Rank of the current process within `num_replicas`.
        :param shuffle: Whether to shuffle the data.
        :param seed: The random seed.
        :param drop_last: Whether to discard samples less than one batch.
        """
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        super().__init__()

    def init_sampler(self, data_source):
        """
        Initialize the sampler with data.

        :param data_source: The data to be sampled.
        """
        return dt.DistributedSampler(
            dataset=data_source,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
