"""Runtime dependency guards for the TensorFlow VAE implementation."""

from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except ImportError as exc:
    TF_AVAILABLE = False
    tf = None
    mixed_precision = None
    TF_IMPORT_ERROR = exc

try:
    import openslide

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    openslide = None

try:
    from PIL import Image, ImageEnhance

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageEnhance = None