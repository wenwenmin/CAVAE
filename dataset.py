import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

class mavc_dataset(Dataset):
    def __init__(self, cli_path, cna_path, rna_path, mic_path, label_path):
        super(mavc_dataset, self).__init__()
        self.cli_data = pd.read_csv(cli_path, index_col=0)
        self.cli_data = torch.tensor(self.cli_data.values, dtype=torch.float32)
        self.cna_data = pd.read_csv(cna_path, index_col=0)
        self.cna_data = torch.tensor(self.cna_data.values, dtype=torch.float32)
        self.rna_data = pd.read_csv(rna_path, index_col=0)
        self.rna_data = torch.tensor(self.rna_data.values, dtype=torch.float32)
        self.mic_data = pd.read_csv(mic_path, index_col=0)
        self.mic_data = torch.tensor(self.mic_data.values, dtype=torch.float32)
        self.label = pd.read_csv(label_path, index_col=0)

        self.x_data = torch.cat((self.cli_data, self.cna_data, self.rna_data, self.mic_data), dim = 1)
        self.y_data = torch.tensor(self.label.values, dtype=torch.float32)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return len(self.x_data)


