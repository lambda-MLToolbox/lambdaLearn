from torch.utils.data.sampler import Sampler


class BaseSampler:
    def __init__(self):
        pass

    def init_sampler(self, data_source):
        """
        Initialize the sampler with data.

        :param data_source: The data to be sampled.
        """
        return Sampler(data_source=data_source)
