import torch_geometric.transforms as gt

from lambdaLearn.Base.Transformer import Transformer


class GDC(Transformer):
    def __init__(
        self,
        self_loop_weight=1,
        normalization_in="sym",
        normalization_out="col",
        diffusion_kwargs=dict(method="ppr", alpha=0.15),
        sparsification_kwargs=dict(method="threshold", avg_degree=64),
        exact=True,
    ):
        """
        :param self_loop_weight: Weight of the added self-loop. Set to None to add no self-loops.
        :param normalization_in: Normalization of the transition matrix on the original (input) graph. Possible values: "sym", "col", and "row".
        :param normalization_out: Normalization of the transition matrix on the transformed GDC (output) graph. Possible values: "sym", "col", and "row".
        :param diffusion_kwargs: Dictionary containing the parameters for diffusion.
        :param sparsification_kwargs: Dictionary containing the parameters for sparsification.
        :param exact: Whether to accurately calculate the diffusion matrix.
        """
        super().__init__()
        self.gdc = gt.GDC(
            self_loop_weight=self_loop_weight,
            normalization_in=normalization_in,
            normalization_out=normalization_out,
            diffusion_kwargs=diffusion_kwargs,
            sparsification_kwargs=sparsification_kwargs,
            exact=exact,
        )

    def transform(self, X):
        X = self.gdc(X)
        return X
