"""CLI: generate images and render the denoising trajectory."""

import argparse
import os

from diffusion.sample import generate_images, load_checkpoint, save_image_grid
from diffusion.visualize import render_denoising_trajectory, render_sample_grid


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained DDPM")
    parser.add_argument("--checkpoint", default="models/ema.pt")
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--digit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--trajectory", action="store_true",
                        help="also render the denoising trajectory")
    args = parser.parse_args()

    model, diffusion, config = load_checkpoint(args.checkpoint)
    os.makedirs(args.out_dir, exist_ok=True)

    images = generate_images(
        model, diffusion, args.n_images, digit=args.digit, seed=args.seed
    )
    grid_path = os.path.join(args.out_dir, "samples.png")
    save_image_grid(images, grid_path)
    print(f"saved {args.n_images} samples -> {grid_path}")

    pretty = os.path.join(args.out_dir, "samples_grid.png")
    render_sample_grid(
        model, diffusion, n_images=args.n_images, digit=args.digit,
        seed=args.seed, save_path=pretty,
    )
    print(f"saved labeled grid -> {pretty}")

    if args.trajectory:
        traj_path = os.path.join(args.out_dir, "denoising_trajectory.png")
        render_denoising_trajectory(
            model, diffusion, digit=args.digit, seed=args.seed, save_path=traj_path
        )
        print(f"saved denoising trajectory -> {traj_path}")


if __name__ == "__main__":
    main()
