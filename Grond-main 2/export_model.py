#!/usr/bin/env python3
"""
Utility to export a trained model pipeline to a versioned file.

This script copies a `.joblib` file (the trained model pipeline) to a
destination directory, appending a version tag to the filename. It
validates the source path, creates the destination directory if it
doesn’t exist, and reports the location of the exported model.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a trained model pipeline to a versioned file "
            "in your models directory."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source joblib file (e.g. output of training or tuning).",
    )
    parser.add_argument(
        "--dest-dir",
        default="models",
        help="Directory where versioned model files will be stored.",
    )
    parser.add_argument(
        "--version",
        default=datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        help="Version tag to append to the filename (default: UTC timestamp).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"ERROR: source file not found at {args.source}", file=sys.stderr)
        raise SystemExit(1)

    os.makedirs(args.dest_dir, exist_ok=True)

    filename = f"xgb_classifier_{args.version}.pipeline.joblib"
    dest_path = os.path.join(args.dest_dir, filename)

    shutil.copy2(args.source, dest_path)
    print(f"Exported model to {dest_path}")


if __name__ == "__main__":
    main()
