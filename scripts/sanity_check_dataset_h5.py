import h5py
import numpy as np

FILES = [
    "data/processed_h5/dataset_train.h5",
    "data/processed_h5/dataset_validation.h5",
    "data/processed_h5/dataset_test.h5",
]

for path in FILES:
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    with h5py.File(path, "r") as f:
        print("Keys:", list(f.keys()))

        for key in f.keys():
            dset = f[key]
            print(f"\n{key}")
            print("  shape:", dset.shape)
            print("  dtype:", dset.dtype)

            # Read only a tiny slice
            try:
                sample = dset[:2]
                print("  sample min:", np.min(sample))
                print("  sample max:", np.max(sample))
                print("  sample mean:", np.mean(sample))
                print("  sample shape:", sample.shape)

                # If this looks like one-hot labels with last dim = 3
                if len(sample.shape) >= 2 and sample.shape[-1] == 3:
                    class_counts = sample.reshape(-1, 3).sum(axis=0)
                    print("  small-sample class counts [non-splice, acceptor, donor]:", class_counts)

                # If this looks like one-hot DNA with last dim = 4
                if len(sample.shape) >= 2 and sample.shape[-1] == 4:
                    base_counts = sample.reshape(-1, 4).sum(axis=0)
                    print("  small-sample base counts [A, C, G, T]:", base_counts)

            except Exception as e:
                print("  could not sample:", repr(e))