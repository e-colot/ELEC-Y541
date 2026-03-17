"""Export AutoEncoder layer weights and biases from mnist_autoencoder.pt into raw binary files.

Generated files (default):
- enc1.weight, enc2.weight, ..., dec4.weight
- enc1.bias, enc2.bias, ..., dec4.bias
- AE_constants.h (C++ header with normalization and scaling constants)

Each file contains float32 values in row-major order.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def _extract_weight_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> list[tuple[int, torch.Tensor]]:
    """Return sorted (layer_index, weight_tensor) tuples for encoder/decoder Linear layers."""
    pattern = re.compile(rf"^{prefix}\.(\d+)\.weight$")
    layers: list[tuple[int, torch.Tensor]] = []

    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            layers.append((int(match.group(1)), value))

    layers.sort(key=lambda item: item[0])
    return layers


def _extract_bias_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> list[tuple[int, torch.Tensor]]:
    """Return sorted (layer_index, bias_tensor) tuples for encoder/decoder Linear layers."""
    pattern = re.compile(rf"^{prefix}\.(\d+)\.bias$")
    layers: list[tuple[int, torch.Tensor]] = []

    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            layers.append((int(match.group(1)), value))

    layers.sort(key=lambda item: item[0])
    return layers


def _generate_constants_header(output_dir: Path) -> None:
    """Generate C++ header with normalization and scaling constants."""
    header_content = """// AE_constants.h
// AutoEncoder normalization and scaling constants
#ifndef _AE_CONSTANTS_H_
#define _AE_CONSTANTS_H_

// Normalization constants (MNIST transform)
#define AE_NORM_MEAN  0.1307f
#define AE_NORM_STD   0.3081f

// Denormalization macro: unnormalized = normalized * std + mean
#define AE_DENORM(x) ((x) * AE_NORM_STD + AE_NORM_MEAN)

// Output scaling: multiply by 255 before casting to uint8
#define AE_SCALE_OUTPUT(x) ((int)((x) * 255.0f) & 0xFF)

#endif // _AE_CONSTANTS_H_
"""
    header_path = output_dir / "AE_constants.h"
    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"Wrote {header_path.name}: normalization and scaling constants")


def export_weights(model_path: Path, output_dir: Path) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")

    # Supports both plain state_dict and checkpoint dictionaries.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Loaded checkpoint is not a valid state_dict dictionary.")

    output_dir.mkdir(parents=True, exist_ok=True)

    enc_weights = _extract_weight_layers(state_dict, "encoder")
    dec_weights = _extract_weight_layers(state_dict, "decoder")
    enc_biases = _extract_bias_layers(state_dict, "encoder")
    dec_biases = _extract_bias_layers(state_dict, "decoder")

    if not enc_weights and not dec_weights:
        raise ValueError("No encoder/decoder weights found in the checkpoint.")

    # Export weights
    for i, (_, weight) in enumerate(enc_weights, start=1):
        out_path = output_dir / f"enc{i}.weight"
        np.asarray(weight.detach().cpu(), dtype=np.float32).tofile(out_path)
        print(f"Wrote {out_path.name}: shape={tuple(weight.shape)}")

    for i, (_, weight) in enumerate(dec_weights, start=1):
        out_path = output_dir / f"dec{i}.weight"
        np.asarray(weight.detach().cpu(), dtype=np.float32).tofile(out_path)
        print(f"Wrote {out_path.name}: shape={tuple(weight.shape)}")

    # Export biases
    for i, (_, bias) in enumerate(enc_biases, start=1):
        out_path = output_dir / f"enc{i}.bias"
        np.asarray(bias.detach().cpu(), dtype=np.float32).tofile(out_path)
        print(f"Wrote {out_path.name}: shape={tuple(bias.shape)}")

    for i, (_, bias) in enumerate(dec_biases, start=1):
        out_path = output_dir / f"dec{i}.bias"
        np.asarray(bias.detach().cpu(), dtype=np.float32).tofile(out_path)
        print(f"Wrote {out_path.name}: shape={tuple(bias.shape)}")

    # Generate C++ header with constants
    _generate_constants_header(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AE layer weights and biases to binary files.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("mnist_autoencoder.pt"),
        help="Path to the trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory where binary weight files will be written.",
    )

    args = parser.parse_args()
    export_weights(args.model, args.out_dir)
