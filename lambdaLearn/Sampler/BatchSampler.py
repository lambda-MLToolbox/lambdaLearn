import torch.utils.data.sampler as torchsampler

from lambdaLearn.Base.BaseSampler import BaseSampler


class BatchSampler(BaseSampler):
    def __init__(self, batch_size: int, drop_last: bool):
        super().__init__()
        """
        :param batch_size: The number of samples in each batch.
        :param drop_last: Whether to discard samples less than one batch.
        """
        self.batch_size = batch_size
        self.drop_last = drop_last

    def init_sampler(self, sampler):
        """
        Initialize batch sampler with sampler.

        :param sampler: The sampler used to initial batch sampler.
        """
        return torchsampler.BatchSampler(sampler=sampler, batch_size=self.batch_size, drop_last=self.drop_last)
