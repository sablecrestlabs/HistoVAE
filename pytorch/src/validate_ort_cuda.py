#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import onnx
import onnxruntime as ort

from .create_onnx import export_histovae_onnx


def summarize_graph(model_path: str) -> None:
    model = onnx.load(model_path)
    ops = Counter(node.op_type for node in model.graph.node)
    print(f"Graph summary for {model_path}:")
    print(f"  total_nodes={len(model.graph.node)}")
    print(
        "  op_counts="
        + ", ".join(f"{name}={count}" for name, count in sorted(ops.items()))
    )


def summarize_profile(profile_path: str) -> None:
    with open(profile_path, encoding="utf-8") as handle:
        events = json.load(handle)

    provider_by_op = defaultdict(Counter)
    provider_by_node_name = defaultdict(Counter)

    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args", {})
        provider = args.get("provider", "unknown")
        op_name = args.get("op_name", "unknown")
        node_name = event.get("name", "")
        provider_by_op[op_name][provider] += 1
        provider_by_node_name[provider][node_name] += 1

    print(f"Profile summary for {profile_path}:")
    for op_name in sorted(provider_by_op):
        counts = provider_by_op[op_name]
        summary = ", ".join(
            f"{provider}={count}" for provider, count in sorted(counts.items())
        )
        print(f"  {op_name}: {summary}")

    conv_summary = provider_by_op.get("Conv", Counter())
    if conv_summary:
        print(
            "  Conv provider breakdown: "
            + ", ".join(
                f"{provider}={count}"
                for provider, count in sorted(conv_summary.items())
            )
        )
    else:
        print("  Conv provider breakdown: no Conv node events found")

    for provider, counts in sorted(provider_by_node_name.items()):
        conv_nodes = [name for name in counts if "conv" in name.lower()]
        if not conv_nodes:
            continue
        print(f"  Example conv-profile events on {provider}:")
        for name in sorted(conv_nodes)[:8]:
            print(f"    {name}")


def print_provider_conclusion(
    available_providers: list[str],
    requested_providers: list[str],
    profile_path: str,
) -> None:
    print("Provider conclusion:")
    print("  available=" + ", ".join(available_providers))
    print("  session=" + ", ".join(requested_providers))

    if "CUDAExecutionProvider" not in available_providers:
        print(
            "  CUDAExecutionProvider is not installed in this ONNX Runtime build, "
            "so every node runs on CPU by necessity."
        )
        print(
            "  This profile does not show a CUDA partitioning failure; it only reflects "
            "that no CUDA provider was available to take any nodes."
        )
        print(
            "  Re-run this script in an environment where ort.get_available_providers() "
            "includes CUDAExecutionProvider to test real CUDA placement."
        )
        return

    if requested_providers and requested_providers[0] == "CUDAExecutionProvider":
        print(
            "  CUDAExecutionProvider was available and requested first. "
            f"Use {profile_path} to inspect actual node placement."
        )
    else:
        print(
            "  CUDAExecutionProvider was available but not first in the session provider list, "
            "so placement may not reflect the intended preference."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HistoVAE to ONNX and profile provider assignment in ONNX Runtime"
    )
    parser.add_argument(
        "--checkpoint",
        default="pretrained/HistoVAE_trained.pt",
        help="Path to the trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--onnx",
        default="tmp_onnx_compare/ort_validate.onnx",
        help="Output ONNX path to export and validate",
    )
    parser.add_argument(
        "--mode",
        choices=["reconstruct", "encode"],
        default="reconstruct",
        help="Which wrapper to export",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used both for static export and for the ORT run input",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch axis instead of the default static batch",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the torch.export-based ONNX exporter",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.onnx) or ".", exist_ok=True)

    export_histovae_onnx(
        checkpoint_path=args.checkpoint,
        out_path=args.onnx,
        mode=args.mode,
        opset=args.opset,
        img_size=args.img_size,
        static_batch=not args.dynamic_batch,
        static_batch_size=args.batch_size if not args.dynamic_batch else None,
        dynamo=args.dynamo,
    )

    summarize_graph(args.onnx)

    available_providers = ort.get_available_providers()
    requested_providers = [
        provider
        for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider in available_providers
    ]
    if not requested_providers:
        raise RuntimeError(
            "No supported ONNX Runtime execution providers are available"
        )

    print("Requested providers: " + ", ".join(requested_providers))
    if "CUDAExecutionProvider" not in requested_providers:
        print(
            "CUDAExecutionProvider is not available in this environment; provider summary will be CPU-only."
        )

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session = ort.InferenceSession(
        args.onnx,
        sess_options=session_options,
        providers=requested_providers,
    )

    input_name = session.get_inputs()[0].name
    input_shape = [
        args.batch_size if isinstance(dim, str) or dim is None else dim
        for dim in session.get_inputs()[0].shape
    ]
    input_tensor = np.random.randn(*input_shape).astype(np.float32)
    session.run(None, {input_name: input_tensor})
    profile_path = session.end_profiling()

    summarize_profile(profile_path)
    print_provider_conclusion(available_providers, requested_providers, profile_path)


if __name__ == "__main__":
    main()
