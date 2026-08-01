import argparse
import h5py
import numpy as np
from pathlib import Path

KEYWORDS = [
    "chr", "chrom", "chromosome", "gene", "name", "transcript",
    "tx", "strand", "coord", "start", "end", "pos", "loc"
]

def decode_value(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x

def sample_dataset(ds, n=10):
    try:
        if ds.shape == ():
            val = ds[()]
            return decode_value(val)
        if len(ds.shape) == 0:
            return decode_value(ds[()])
        take = min(n, ds.shape[0])
        vals = ds[:take]
        if vals.dtype.kind in {"S", "O"}:
            vals = [decode_value(v) for v in vals]
        return vals
    except Exception as e:
        return f"<could not sample: {e}>"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--max-items", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.h5)
    print(f"\nInspecting: {path}")
    print(f"Exists: {path.exists()}")
    print(f"Size: {path.stat().st_size / (1024**3):.3f} GB\n")

    count = 0

    with h5py.File(path, "r") as h5:
        print("Root attributes:")
        for k, v in h5.attrs.items():
            print(f"  {k}: {v}")

        print("\nTop-level keys:")
        for k in h5.keys():
            obj = h5[k]
            if isinstance(obj, h5py.Dataset):
                print(f"  {k}: DATASET shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {k}: GROUP")

        print("\nPotential metadata datasets:")
        def visit(name, obj):
            nonlocal count
            if count >= args.max_items:
                return

            lname = name.lower()
            is_interesting = any(kw in lname for kw in KEYWORDS)

            if isinstance(obj, h5py.Dataset):
                if is_interesting or obj.dtype.kind in {"S", "O", "U"}:
                    count += 1
                    print("\n---")
                    print(f"name: {name}")
                    print(f"shape: {obj.shape}")
                    print(f"dtype: {obj.dtype}")
                    for ak, av in obj.attrs.items():
                        print(f"attr {ak}: {av}")
                    print("sample:", sample_dataset(obj))

            elif isinstance(obj, h5py.Group):
                if is_interesting:
                    count += 1
                    print("\n---")
                    print(f"group: {name}")
                    for ak, av in obj.attrs.items():
                        print(f"attr {ak}: {av}")

        h5.visititems(visit)

    print(f"\nPrinted {count} potential metadata items.")

if __name__ == "__main__":
    main()
