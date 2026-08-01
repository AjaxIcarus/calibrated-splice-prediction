import argparse
from pathlib import Path

import h5py


def make_tiny_h5(input_path, output_path, groups=4, segments_per_group=16):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        x_keys = sorted(
            [k for k in src.keys() if k.startswith("X")],
            key=lambda x: int(x[1:])
        )

        kept = 0

        for old_x_key in x_keys[:groups]:
            old_idx = old_x_key[1:]
            old_y_key = f"Y{old_idx}"

            if old_y_key not in src:
                print(f"Skipping {old_x_key}; no matching {old_y_key}")
                continue

            x = src[old_x_key]
            y = src[old_y_key]

            n = min(segments_per_group, x.shape[0], y.shape[1])

            new_x_key = f"X{kept}"
            new_y_key = f"Y{kept}"

            print(f"{old_x_key}/{old_y_key} -> {new_x_key}/{new_y_key}, n={n}")

            dst.create_dataset(new_x_key, data=x[:n], dtype=x.dtype)
            dst.create_dataset(new_y_key, data=y[:, :n, :, :], dtype=y.dtype)

            kept += 1

    print(f"\nWrote tiny file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--segments-per-group", type=int, default=16)
    args = parser.parse_args()

    make_tiny_h5(
        input_path=args.input,
        output_path=args.output,
        groups=args.groups,
        segments_per_group=args.segments_per_group,
    )