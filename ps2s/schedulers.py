from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def build_NoamLike_LRSched(optimizer: Optimizer, warmup_steps: int, cooldown_steps: int, init_lr: float, max_lr: float, final_lr: float):

    def lr_lambda(step: int):
        if step < warmup_steps:
            warmup_factor = (max_lr - init_lr) / warmup_steps
            return step * warmup_factor / init_lr + 1
        elif warmup_steps <= step < warmup_steps + cooldown_steps:
            cooldown_factor = (final_lr / max_lr) ** (1 / cooldown_steps)
            return max_lr * cooldown_factor ** (step - warmup_steps) / init_lr
        else:
            return final_lr / init_lr
    return LambdaLR(optimizer, lr_lambda)
