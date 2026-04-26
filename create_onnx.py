#!/usr/bin/env python3

import argparse

import torch
import torch.nn as nn

from vae import VAE, VAEConfig


class HistoVAEReconstructONNX(nn.Module):
    """
    ONNX-friendly wrapper.

    Input:
        x: float32 NCHW tensor, normalized to [-1, 1]
           shape: [B, 3, 256, 256]

    Output:
        x_recon: float32 NCHW tensor, normalized to [-1, 1]
                 shape: [B, 3, 256, 256]
    """

    def __init__(self, model: VAE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic reconstruction: use mu, not random sampling.
        mu, _logvar = self.model.encode(x)
        z = torch.clamp(mu, min=-10.0, max=10.0)
        x_recon = self.model.decode(z)
        return torch.clamp(x_recon, min=-1.0, max=1.0)


class HistoVAEEncodeONNX(nn.Module):
    """
    Optional encoder-only wrapper.

    Input:
        x: [B, 3, 256, 256] in [-1, 1]

    Outputs:
        mu:     [B, latent_channels, latent_h, latent_w]
        logvar: [B, latent_channels, latent_h, latent_w]
    """

    def __init__(self, model: VAE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model.encode(x)


def load_histovae(checkpoint_path: str) -> VAE:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    cfg = ckpt.get("config", {})

    config = VAEConfig(
        img_channels=cfg.get("img_channels", 3),
        img_size=cfg.get("img_size", 256),
        base_channels=cfg.get("base_channels", 32),
        channel_multipliers=tuple(cfg.get("channel_multipliers", (1, 2, 4))),
        latent_channels=cfg.get("latent_channels", 32),

        # Important: these were not saved in older checkpoints, but they affect
        # architecture. Match the training defaults from vae.py.
        num_res_blocks_per_stage=cfg.get("num_res_blocks_per_stage", 2),
        use_attention_at=tuple(cfg.get("use_attention_at", (32,))),
    )

    model = VAE(config=config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def export_histovae_onnx(
    checkpoint_path: str,
    out_path: str,
    mode: str = "reconstruct",
    opset: int = 18,
    img_size: int | None = None,
    static_batch: bool = True,
    static_batch_size: int | None = 1,
    dynamo: bool = False,
) -> None:
    if static_batch_size is not None and static_batch_size < 1:
        raise ValueError("static_batch_size must be >= 1")
    if static_batch_size is not None and not static_batch:
        raise ValueError("static_batch_size requires static_batch=True")

    model = load_histovae(checkpoint_path)

    export_batch_size = static_batch_size or 1
    resolved_img_size = img_size or model.img_size
    dummy = torch.randn(
        export_batch_size,
        model.in_channels,
        resolved_img_size,
        resolved_img_size,
        dtype=torch.float32,
    )

    if mode == "reconstruct":
        wrapper = HistoVAEReconstructONNX(model).eval()
        output_names = ["reconstruction"]
    else:
        wrapper = HistoVAEEncodeONNX(model).eval()
        output_names = ["mu", "logvar"]

    dynamic_axes = None
    if not static_batch:
        if mode == "reconstruct":
            dynamic_axes = {
                "input": {0: "batch"},
                "reconstruction": {0: "batch"},
            }
        else:
            dynamic_axes = {
                "input": {0: "batch"},
                "mu": {0: "batch"},
                "logvar": {0: "batch"},
            }

    export_kwargs = {
        "input_names": ["input"],
        "output_names": output_names,
        "opset_version": opset,
        "export_params": True,
        "do_constant_folding": True,
        "dynamo": dynamo,
    }
    if dynamic_axes is not None:
        export_kwargs["dynamic_axes"] = dynamic_axes

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            **export_kwargs,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="pretrained/HistoVAE_trained.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--out",
        default="pretrained/HistoVae_reconstruct.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--mode",
        choices=["reconstruct", "encode"],
        default="reconstruct",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument(
        "--static-batch-size",
        type=int,
        default=None,
        help="Bake a fixed batch size into the exported ONNX graph",
    )
    parser.add_argument(
        "--static-batch",
        dest="static_batch",
        action="store_true",
        help="Export with a fully static batch dimension",
    )
    parser.add_argument(
        "--dynamic-batch",
        dest="static_batch",
        action="store_false",
        help="Export with a dynamic batch axis instead of the CUDA-friendlier static default",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the newer torch.export-based ONNX exporter",
    )
    parser.set_defaults(static_batch=None)
    args = parser.parse_args()

    if args.static_batch is None:
        args.static_batch = True
    if args.static_batch and args.static_batch_size is None:
        args.static_batch_size = 1

    export_histovae_onnx(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        mode=args.mode,
        opset=args.opset,
        img_size=args.img_size,
        static_batch=args.static_batch,
        static_batch_size=args.static_batch_size,
        dynamo=args.dynamo,
    )

    batch_mode = (
        f"static batch={args.static_batch_size or 1}"
        if args.static_batch
        else "dynamic batch"
    )
    exporter = "dynamo" if args.dynamo else "legacy"
    print(
        f"Exported {args.mode} model to {args.out} "
        f"({batch_mode}, exporter={exporter})"
    )
    if not args.static_batch:
        print(
            "Note: dynamic batch keeps Shape-driven GroupNorm restore paths in the ONNX graph, "
            "which can reduce ORT CUDA partitioning compared with the default static-batch export."
        )


if __name__ == "__main__":
    main()