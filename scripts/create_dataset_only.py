from types import SimpleNamespace
from openspliceai.create_data import create_dataset

args = SimpleNamespace(
    output_dir="data/processed_h5/",
    flanking_size=80,
    verify_h5=True,
    chr_split="train-test",
    biotype="protein-coding"
)

create_dataset.create_dataset(args)