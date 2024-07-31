import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix,roc_curve,auc,recall_score, precision_score,accuracy_score,matthews_corrcoef, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import random_split
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from model import MAVC
from dataset import mavc_dataset
from torchstat import stat


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Set seed
set_seed(3407)
#Dataset
#train
cli_train_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/train/brca_clinical.csv'
cna_train_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/train/brca_cna.csv'
rna_train_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/train/brca_rna_exp.csv'
mic_train_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/train/brca_micro.csv'
label_train_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/train/brca_label.csv'

#valid
cli_valid_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/valid/brca_clinical.csv'
cna_valid_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/valid/brca_cna.csv'
rna_valid_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/valid/brca_rna_exp.csv'
mic_valid_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/valid/brca_micro.csv'
label_valid_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/valid/brca_label.csv'
#test
cli_test_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/test/brca_clinical.csv'
cna_test_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/test/brca_cna.csv'
rna_test_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/test/brca_rna_exp.csv'
mic_test_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/test/brca_micro.csv'
label_test_data = '/data/minwenwen/lixiaoyu/TCGA/data/BRCA/test/brca_label.csv'

train_data = mavc_dataset(cli_train_data, cna_train_data, rna_train_data, mic_train_data, label_train_data)
valid_data = mavc_dataset(cli_valid_data, cna_valid_data, rna_valid_data, mic_valid_data, label_valid_data)
test_data = mavc_dataset(cli_test_data, cna_test_data, rna_test_data, mic_test_data, label_test_data)

train_data_size = len(train_data)
valid_data_size = len(valid_data)
test_data_size = len(test_data)
print("Training device:{}".format(torch.cuda.get_device_name()))
print("Train dataset length：{}".format(train_data_size))
print("Valid dataset length：{}".format(valid_data_size))
print("Test dataset length：{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

#Model
dropout = 0.895
BN = 1e-3
num_head = 64
K = 64
V = 64
model = MAVC(n_head_=num_head, d_k_=K, d_v_=V, drop_out = dropout, batch_norml = BN)
model.to(device)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.05, patience=50, verbose=False)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
num_epoch = 1000
max_accuracy = 0
mavc_acc = 0
mavc_pre = 0
mavc_sen = 0
mavc_f1s = 0
train_loss = 0
total_train_step = 0
total_test_step = 0

#Train
loop = tqdm(range(num_epoch), total=num_epoch)
for epoch in loop:
    total_valid_loss = 0
    total_accuracy = 0
    model.train()
    # loop = tqdm((train_dataloader), total=len(train_dataloader))
    for data in train_dataloader:
        x_data, y_data = data
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        y_data = y_data.squeeze(dim = 1)
        y_pred = model(x_data)
        train_loss = criterion(y_pred, y_data.long())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step(train_loss)
        loop.set_description(f'Epoch [{epoch}/{num_epoch}]')
        total_valid_loss = total_valid_loss + train_loss.item()
        accuracy = (y_pred.argmax(1) == y_data).sum()
        total_accuracy = total_accuracy + accuracy
        if ((total_accuracy / len(train_data)) > max_accuracy):
            max_accuracy = total_accuracy / len(train_data)
            torch.save(model, "/data/minwenwen/lixiaoyu/TCGA/Model/MAVC_Best.pth")

    model.eval()
    with torch.no_grad():
        for data in valid_dataloader:
            x_data, y_data = data
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            y_data = y_data.squeeze(dim=1)
            y_pred = model(x_data)
            valid_loss = criterion(y_pred, y_data.long())
            loop.set_postfix(train_loss=train_loss.item(), valid_loss=valid_loss.item(), learning_rate=learning_rate)

model = torch.load('/data/minwenwen/lixiaoyu/TCGA/Model/MAVC_Best.pth')
model.eval()
with torch.no_grad():
    for data in test_dataloader:
        mavc_x_data, mavc_y_data = data
        mavc_x_data = mavc_x_data.to(device)
        mavc_y_data = mavc_y_data.to(device)
        mavc_y_pred = model(mavc_x_data)

        mavc_y_data = mavc_y_data.squeeze(dim=1)

        mavc_y_data = mavc_y_data.detach().cpu().numpy()
        mavc_y_pred = mavc_y_pred.detach().cpu().numpy()
        mavc_acc = accuracy_score(mavc_y_data, mavc_y_pred.argmax(1))
        mavc_pre = precision_score(mavc_y_data, mavc_y_pred.argmin(1))
        mavc_sen = recall_score(mavc_y_data, mavc_y_pred.argmin(1))
        mavc_f1s = f1_score(mavc_y_data, mavc_y_pred.argmin(1))

        mavc_y_pred = np.amin(mavc_y_pred, axis=1)

# print(mavc_y_pred)
# print(mavc_y_data)
# y_data = y_data.detach().cpu().numpy()
# y_pred = y_pred.detach().cpu().numpy()
# y_pred = np.amin(y_pred, axis=1)
mavc_auc = roc_auc_score(mavc_y_data, mavc_y_pred)
print("MAVC's accuracy:{:.3f}".format(mavc_acc))
print("MAVC's precision:{:.3f}".format(mavc_pre))
print("MAVC's sensitivity:{:.3f}".format(mavc_sen))
print("MAVC's f1score:{:.3f}".format(mavc_f1s))
print("MAVC's AUC:{:.3f}".format(mavc_auc))






