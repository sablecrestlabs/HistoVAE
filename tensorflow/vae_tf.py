#!/usr/bin/env python3
"""TensorFlow VAE training entrypoint.

The implementation is split across sibling modules in this directory so the
model, data pipeline, losses, and training orchestration can evolve
independently while this script remains compatible with existing launchers.
"""

from vae_tf_cli import main


if __name__ == "__main__":
    main()
