import sys
import torch
import numpy as np
import pandas as pd
import sklearn
import h5py
import Bio

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("h5py:", h5py.__version__)
print("Biopython imported successfully")