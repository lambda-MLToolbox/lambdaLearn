from torch.utils.data import sampler

from lambdaLearn.Base.BaseSampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, replacement: bool = False, num_samples=None, generator=None):
        """
        :param replacement: samples are drawn on-demand with replacement if True.
        :param num_samples: The number of samples
        :param generator: Generator used in sampling.
        """

        super().__init__()
        self.replacement = replacement
        self.num_samples = num_samples
        self.generator = generator

    def init_sampler(self, data_source):
        """
        Initialize the sampler with data.

        :param data_source: The data to be sampled.
        """
        return sampler.RandomSampler(
            data_source=data_source,
            replacement=self.replacement,
            num_samples=self.num_samples,
            generator=self.generator,
        )
