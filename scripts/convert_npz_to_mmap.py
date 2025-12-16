#!/usr/bin/env python
"""
Convert existing npz MLX shards to memory-mapped npy (uint16) pairs.
Optionally deletes the old npz/mlx files after successful conversion.
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def convert_npz_shard(npz_path: Path, delete_old: bool = False):
    data = np.load(npz_path)
    x_arr = data["x"].astype(np.uint16)
    y_arr = data["y"].astype(np.uint16)
    out_x = npz_path.with_name(npz_path.stem + "_x.npy")
    out_y = npz_path.with_name(npz_path.stem + "_y.npy")
    np.save(out_x, x_arr)
    np.save(out_y, y_arr)
    if delete_old:
        npz_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Convert npz shards to mmap-friendly npy pairs.")
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory containing .npz shards")
    parser.add_argument("--delete-old", action="store_true", help="Delete npz files after conversion")
    args = parser.parse_args()

    shard_dir = args.shard_dir
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"No npz shards found in {shard_dir}")

    print(f"Converting {len(shards)} shards in {shard_dir} to npy (uint16)...")
    for shard in tqdm(shards):
        convert_npz_shard(shard, delete_old=args.delete_old)

    if args.delete_old:
        mlx_files = list(shard_dir.glob("shard_*.mlx"))
        for f in mlx_files:
            f.unlink()
        print("Old npz/mlx files removed.")
    print("✅ Conversion complete.")


if __name__ == "__main__":
    main()
