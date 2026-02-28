#!/usr/bin/env python3
"""
Export NAFNet PyTorch model to ONNX format for Rust integration.
Self-contained version that doesn't require BasicSR installation.

Usage:
    python export_nafnet.py [--width 32|64] [--fp16] [--output ./models]

Example:
    python export_nafnet.py --width 64 --fp16 --output ./models

This will download NAFNet-SIDD-width64 pretrained weights, export to ONNX,
optionally convert to FP16, and output the model file + SHA256 hash.
"""

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Please install: pip install torch")
    sys.exit(1)

# Model URLs from megvii-research/NAFNet releases
MODEL_URLS = {
    "width32": "https://github.com/megvii-research/NAFNet/releases/download/v0.1.0/NAFNet-SIDD-width32.pth",
    "width64": "https://github.com/megvii-research/NAFNet/releases/download/v0.1.0/NAFNet-SIDD-width64.pth",
}


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        # Match the original BasicSR implementation exactly
        # Sequential: 0=pool, 1=conv, 2=sigmoid
        self.add_module("0", nn.AdaptiveAvgPool2d(1))
        self.add_module("1", nn.Conv2d(channels, channels, 1, bias=True))
        self.add_module("2", nn.Sigmoid())
        self.channels = channels

    def forward(self, x):
        # We need to override forward to implement x * attention
        w = super().forward(x)
        return x * w


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        self.sg = SimpleGate()
        self.sca = SimpleChannelAttention(dw_channel // 2)
        self.norm1 = LayerNorm2d(c)

        # FFN
        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x

        # Local attention (first branch)
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        # FFN (second branch)
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x


def download_file(url: str, dest: Path) -> None:
    """Download file from URL to destination path."""
    import urllib.request
    import ssl

    print(f"Downloading {url}...")

    # Create SSL context that doesn't verify certificates (some systems have issues)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
        print(
            f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)",
            end="",
            flush=True,
        )

    urllib.request.urlretrieve(url, dest, reporthook=report_progress)
    print()  # Newline after progress


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_nafnet_model(width: int, checkpoint_path: Path) -> NAFNet:
    """Load NAFNet model with given width from checkpoint."""
    model = NAFNet(
        img_channel=3,
        width=width,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle both direct state_dict and wrapped checkpoint formats
    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(
    model: NAFNet,
    output_path: Path,
    opset_version: int = 14,
) -> None:
    """Export PyTorch model to ONNX format."""
    print(f"Exporting to ONNX (opset {opset_version})...")

    # Create dummy input with batch dimension
    # NAFNet accepts variable input sizes, but must be divisible by 16
    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"Exported to: {output_path}")


def simplify_onnx(input_path: Path, output_path: Path) -> None:
    """Simplify ONNX model using onnxsim."""
    try:
        import onnx
        from onnxsim import simplify

        print("Simplifying ONNX model...")
        model = onnx.load(str(input_path))
        model_simp, check = simplify(model)

        if check:
            onnx.save(model_simp, str(output_path))
            print(f"Simplified model saved to: {output_path}")
        else:
            print("Warning: Simplification check failed, using original model")
            os.rename(input_path, output_path)
    except ImportError:
        print("Warning: onnxsim not installed, skipping simplification")
        os.rename(input_path, output_path)


def convert_to_fp16(input_path: Path, output_path: Path) -> None:
    """Convert ONNX model to FP16 using onnxconverter-common."""
    try:
        from onnxconverter_common.float16 import convert_float_to_float16
        import onnx

        print("Converting to FP16...")
        model = onnx.load(str(input_path))
        model_fp16 = convert_float_to_float16(model)
        onnx.save(model_fp16, str(output_path))
        print(f"FP16 model saved to: {output_path}")
    except ImportError:
        print("Warning: onnxconverter-common not installed, skipping FP16 conversion")
        print("Install with: pip install onnxconverter-common")
        os.rename(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export NAFNet model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --width 64                    # Export width-64 model
  %(prog)s --width 64 --fp16             # Export and convert to FP16
  %(prog)s --width 32 --output ./models  # Export to specific directory
""",
    )

    parser.add_argument(
        "--width",
        type=int,
        choices=[32, 64],
        default=64,
        help="Model width (32 or 64). Width-64 has better quality but larger size.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert model to FP16 (half precision). Reduces size by 50%%.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Output directory for the model file (default: current directory)",
    )
    parser.add_argument(
        "--keep-weights",
        action="store_true",
        help="Keep downloaded PyTorch weights file after export",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Download pretrained weights
    width_key = f"width{args.width}"
    weights_url = MODEL_URLS[width_key]
    weights_filename = Path(weights_url).name
    weights_path = args.output / weights_filename

    if not weights_path.exists():
        print(f"Downloading NAFNet-SIDD-{width_key} pretrained weights...")
        download_file(weights_url, weights_path)
        print(f"Saved weights to: {weights_path}")
    else:
        print(f"Using existing weights: {weights_path}")

    # Load model
    print(f"\nLoading NAFNet model (width={args.width})...")
    model = load_nafnet_model(args.width, weights_path)
    print("Model loaded successfully")

    # Export to ONNX
    temp_onnx_path = args.output / f"nafnet-sidd-{width_key}-temp.onnx"
    export_to_onnx(model, temp_onnx_path)

    # Simplify
    simplified_path = args.output / f"nafnet-sidd-{width_key}-simplified.onnx"
    simplify_onnx(temp_onnx_path, simplified_path)
    if temp_onnx_path.exists():
        os.remove(temp_onnx_path)

    # Optionally convert to FP16
    if args.fp16:
        final_path = args.output / f"nafnet-sidd-{width_key}-fp16.onnx"
        convert_to_fp16(simplified_path, final_path)
        os.remove(simplified_path)
    else:
        final_path = args.output / f"nafnet-sidd-{width_key}.onnx"
        os.rename(simplified_path, final_path)

    # Compute SHA256
    print("\nComputing SHA256 hash...")
    sha256_hash = compute_sha256(final_path)

    # Print summary
    file_size = final_path.stat().st_size
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Model file: {final_path}")
    print(f"File size:  {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)")
    print(f"SHA256:     {sha256_hash}")
    print("=" * 60)
    print("\nFor Rust integration, add these constants to your code:")
    print(f'  const NAFNET_FILENAME: &str = "{final_path.name}";')
    print(f'  const NAFNET_SHA256: &str = "{sha256_hash}";')
    print(f"  // Upload to GitHub Releases and use URL:")
    print(
        f"  // https://github.com/YOUR_USER/YOUR_REPO/releases/download/TAG/{final_path.name}"
    )

    # Clean up weights if not keeping
    if not args.keep_weights:
        print(f"\nRemoving weights file: {weights_path}")
        os.remove(weights_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
