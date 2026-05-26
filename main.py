import argparse
import os
import torch

from config import (
    DEVICE, MAX_SIZE, NUM_STEPS, STYLE_WEIGHT,
    CONTENT_WEIGHT, TV_WEIGHT, STYLE_WEIGHTS,
    SAVE_EVERY, OUTPUT_DIR
)
from src.utils import (
    load_image, show_images,
    show_output, show_intermediates, save_image
)
from src.model import build_vgg_base, build_model
from src.transfer import run_style_transfer


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Style Transfer using VGG19")

    parser.add_argument(
        "--content", type=str, required=True,
        help="Path to the content image (e.g. images/content/photo.jpg)"
    )
    parser.add_argument(
        "--style", type=str, required=True,
        help="Path to the style image (e.g. images/style/painting.jpg)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the output image (default: outputs/result.jpg)"
    )
    parser.add_argument(
        "--steps", type=int, default=NUM_STEPS,
        help=f"Number of optimization steps (default: {NUM_STEPS})"
    )
    parser.add_argument(
        "--style-weight", type=float, default=STYLE_WEIGHT,
        help=f"Style loss weight (default: {STYLE_WEIGHT})"
    )
    parser.add_argument(
        "--content-weight", type=float, default=CONTENT_WEIGHT,
        help=f"Content loss weight (default: {CONTENT_WEIGHT})"
    )
    parser.add_argument(
        "--tv-weight", type=float, default=TV_WEIGHT,
        help=f"Total variation loss weight (default: {TV_WEIGHT})"
    )
    parser.add_argument(
        "--max-size", type=int, default=MAX_SIZE,
        help=f"Max image size in pixels (default: {MAX_SIZE})"
    )
    parser.add_argument(
        "--no-intermediates", action="store_true",
        help="Skip saving and showing intermediate outputs"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = args.output if args.output else os.path.join(OUTPUT_DIR, "result.jpg")

    # ── Load Images ────────────────────────────────────────────────────────────
    print(f"Loading content image: {args.content}")
    print(f"Loading style image  : {args.style}")

    content_tensor = load_image(args.content, max_size=args.max_size, device=DEVICE)
    style_tensor   = load_image(args.style,   max_size=args.max_size, device=DEVICE)

    show_images(content_tensor, style_tensor)

    # ── Build Model ────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    cnn = build_vgg_base(DEVICE)

    model, content_losses, style_losses, tv_loss = build_model(
        cnn=cnn,
        content_img=content_tensor,
        style_img=style_tensor,
        style_weights=STYLE_WEIGHTS,
        tv_weight=args.tv_weight,
        device=DEVICE
    )

    # ── Run Optimization ───────────────────────────────────────────────────────
    input_tensor = content_tensor.clone()

    output, intermediates = run_style_transfer(
        model=model,
        content_losses=content_losses,
        style_losses=style_losses,
        tv_loss=tv_loss,
        input_img=input_tensor,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        save_every=SAVE_EVERY,
    )

    # ── Show and Save Results ──────────────────────────────────────────────────
    if not args.no_intermediates and intermediates:
        show_intermediates(intermediates)

    show_output(output)
    save_image(output, output_path)


if __name__ == "__main__":
    main()