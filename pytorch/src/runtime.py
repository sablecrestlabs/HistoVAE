"""Runtime dependency guards for the PyTorch VAE implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import GradScaler, autocast

    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    GradScaler = None
    autocast = None

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    from torchvision import transforms
    from torchvision.utils import make_grid

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    transforms = None
    make_grid = None

try:
    import openslide

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    openslide = None

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
