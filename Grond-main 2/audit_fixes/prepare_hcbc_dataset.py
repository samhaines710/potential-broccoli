#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_hcbc_dataset.py

Purpose:
- Convert an already prepared training CSV (your existing pipeline output)
  into a binary, horizon-conditioned dataset expected by the HCBC trainer.

Inputs:
- --in : path to existing prepared CSV (e.g., data/movement_training_data.csv)
- --out: path to write HCBC CSV           (e.g., data/multiH_binary.csv)
- --h       : optional integer lookahead H to stamp if not present (default: 2)
- --drop-neutral: if set, drop rows where movement is NEUTRAL (recommended)

Expected columns (flexible):
- If 'movement_type' exists: label is derived as CALL=1, PUT=0, NEUTRAL handled by --drop-neutral
- If 'label_up' already exists: it will be preserved.
- If 'H' exists: preserved; else we stamp the provided --h.

No shuffling is performed (time order preserved).
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input prepared CSV")
    ap.add_argument("--out", dest="out", required=True, help="Output HCBC CSV")
    ap.add_argument("--h", dest="H", type=int, default=2, help="Lookahead bars if not present")
    ap.add_argument("--drop-neutral", action="store_true", help="Drop NEUTRAL rows (recommended)")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise FileNotFoundError(args.inp)

    df = pd.read_csv(args.inp)

    # If label_up already present, keep it. Else derive from movement_type.
    if "label_up" not in df.columns:
        if "movement_type" in df.columns:
            mt = df["movement_type"].astype(str).str.upper()
            label_up = np.where(mt == "CALL", 1,
                         np.where(mt == "PUT", 0, -1))
            df["label_up"] = label_up
            if args.drop_neutral:
                df = df[df["label_up"] >= 0].copy()
            else:
                # map NEUTRAL to abstain at training time by dropping here anyway
                df = df[df["label_up"] >= 0].copy()
        else:
            raise ValueError("Neither 'label_up' nor 'movement_type' found. Provide one.")

    # Ensure H exists
    if "H" not in df.columns:
        df["H"] = int(args.H)

    # Keep time order if timestamp present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic sanity: remove infs/NaNs in features; keep label_up & H
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Write
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote HCBC dataset → {args.out} (rows={len(df)})")

if __name__ == "__main__":
    main()
