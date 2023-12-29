from torch.utils.data import sampler

from lambdaLearn.Base.BaseSampler import BaseSampler


class SequentialSampler(BaseSampler):
    def __init__(self):
        super().__init__()

    def init_sampler(self, data_source):
        """
        Initialize the sampler with data.

        :param data_source: The data to be sampled.
        """
        return sampler.SequentialSampler(data_source=data_source)
