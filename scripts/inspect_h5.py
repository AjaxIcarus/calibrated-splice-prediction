import h5py
from pathlib import Path

FILES = [
    "data/processed_h5/datafile_train.h5",
    "data/processed_h5/datafile_validation.h5",
    "data/processed_h5/datafile_test.h5",
    "data/processed_h5/dataset_train.h5",
    "data/processed_h5/dataset_validation.h5",
    "data/processed_h5/dataset_test.h5",
]

def describe_dataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}")
        print(f"  shape: {obj.shape}")
        print(f"  dtype: {obj.dtype}")
        print(f"  chunks: {obj.chunks}")
        print(f"  compression: {obj.compression}")
    elif isinstance(obj, h5py.Group):
        print(f"{name}/")

for file_path in FILES:
    path = Path(file_path)

    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    if not path.exists():
        print("MISSING")
        continue

    print(f"size: {path.stat().st_size / (1024**3):.2f} GB")

    with h5py.File(path, "r") as f:
        print("\nTop-level keys:")
        for key in f.keys():
            print(f"  - {key}")

        print("\nFull structure:")
        f.visititems(describe_dataset)

        print("\nAttributes:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")
