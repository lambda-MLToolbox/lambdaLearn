import torch
from torchtext import vocab

from lambdaLearn.Base.Transformer import Transformer


class FastText(Transformer):
    def __init__(
        self,
        language="en",
        lower_case_backup=True,
        unk_init=None,
        pad_init=None,
        pad_token="<pad>",
        unk_token="<unk>",
        cache=None,
    ):
        """
        :param language: Language type.
        :param lower_case_backup: Whether to convert all to lowercase when looking up words.
        :param unk_init: By default, initialize out-of-vocabulary word vectors to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size.
        :param pad_init: By default, initialize out-of-vocabulary word vectors to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size.
        :param pad_token: The default padding token.
        :param unk_token: The default token represents unknown words.
        """
        super().__init__()
        self.vec = vocab.FastText(language=language, cache=cache)
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.pad_init = torch.Tensor.zero_ if pad_init is None else pad_init
        self.lower_case_backup = lower_case_backup
        self.vec.stoi[pad_token] = self.vec.vectors.shape[0]
        self.vec.stoi[unk_token] = self.vec.vectors.shape[0] + 1
        self.vec.vectors = torch.cat(
            [
                self.vec.vectors,
                self.pad_init(torch.Tensor(self.vec.vectors.shape[1])),
                self.unk_init(torch.Tensor(self.vec.vectors.shape[1])),
            ],
            dim=0,
        )

    def transform(self, X):
        if isinstance(X, tuple):
            X = list(X)
        if isinstance(X, list):
            return self.vec.get_vecs_by_tokens(X, lower_case_backup=self.lower_case_backup)
        else:
            return self.vec[X]
