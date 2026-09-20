"""CLI: train the diffusion model on MNIST."""

import argparse
import os

from diffusion.train import DEFAULTS, dry_run, train


def main():
    parser = argparse.ArgumentParser(description="Train the DDPM on MNIST")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--timesteps", type=int, default=DEFAULTS["timesteps"])
    parser.add_argument("--subset", type=int, default=DEFAULTS["subset"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--base-channels", type=int, default=DEFAULTS["base_channels"])
    parser.add_argument(
        "--schedule", choices=["linear", "cosine"], default=DEFAULTS["schedule"]
    )
    parser.add_argument("--conditional", action="store_true",
                        help="enable class-conditional generation")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    parser.add_argument("--data-dir", default=DEFAULTS["data_dir"])
    parser.add_argument("--dry-run", action="store_true",
                        help="2 synthetic steps, no dataset download")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <out-dir>/last.pt if it exists")
    parser.add_argument("--reset-ema", action="store_true",
                        help="with --resume, re-seed EMA from loaded weights")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return
    resume_from = os.path.join(args.out_dir, "last.pt") if args.resume else None
    try:
        train(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "timesteps": args.timesteps,
            "subset": args.subset,
            "lr": args.lr,
            "base_channels": args.base_channels,
            "schedule": args.schedule,
            "conditional": args.conditional,
            "seed": args.seed,
            "out_dir": args.out_dir,
            "data_dir": args.data_dir,
        },
        resume_from=resume_from,
        reset_ema=args.reset_ema,
        )
    except Exception:
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
