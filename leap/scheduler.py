from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    ConstantLR,
    OneCycleLR,
    StepLR,
)
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
