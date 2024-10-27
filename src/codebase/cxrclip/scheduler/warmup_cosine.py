import math
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupCosineAnnealingLR(LambdaLR):
    r"""
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it to zero by cosine decay. After linear warmup,
    the LR decays as:
    .. math::
        \eta_t = \eta_{max}\cos^2(\frac{T_{cur} - T_{warm}}{T_{max} - T_{warm}}\frac{\pi}{2})
    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper optimizer.
    total_steps: int
        Total epochs (or iterations) for training.
    warmup_steps: int
        Number of first few steps to do linear warmup.
    last_epoch: int, optional (default = -1)
        The index of last step (epoch or iteration). We named it ``last_epoch``
        instead of ``last_step`` to keep the naming consistent with other LR
        schedulers in PyTorch.
    """

    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: Union[int, float], last_epoch: int = -1, **kwargs):
        assert warmup_steps < total_steps, "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        if isinstance(warmup_steps, float):
            self.wsteps = math.ceil(total_steps * warmup_steps)
        else:
            self.wsteps = warmup_steps

        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Cosine annealing decay.
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        # Avoid negative learning rate.
        return max(0, multiplier)
