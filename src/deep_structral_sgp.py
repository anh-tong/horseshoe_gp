import torch
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor
from gpytorch.models import ApproximateGP
from torch.distributions import Normal

from src.structural_sgp import StructuralSparseGP


class DeepStructuralLayer(ApproximateGP):

    def __init__(self, input_dims, output_dims, gp: StructuralSparseGP):
        super().__init__(None)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.gp = gp
        self.variational_strategy = gp.variational_strategy

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, inputs, are_samples=False, **kwargs):
        is_deterministic = not are_samples
        if isinstance(inputs, MultitaskMultivariateNormal):
            inputs = Normal(loc=inputs.mean, scale=inputs.variance.sqrt()).rsample()
            is_deterministic = False

        if self.output_dims is not None:
            inputs = inputs.unsqueeze(-3)
            inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])

        # the main different with the normal DeepGPLayer
        output = self.gp.__call__(inputs)

        if self.output_dims is not None:
            mean = output.loc.transpose(-1, -2)
            covar = BlockDiagLazyTensor(output.lazy_covariance_matrix, block_dim=-3)
            output = MultitaskMultivariateNormal(mean, covar, interleaved=False)

        if is_deterministic:
            output = output.expand(torch.Size([settings.num_likelihood_samples.value()]) + output.batch_shape)

        return output