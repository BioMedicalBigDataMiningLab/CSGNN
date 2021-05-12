import torch
import numpy as np

from parms_setting import settings
from data_preprocess import load_data
from instantiation import Create_model
from train import train_model


# parameters setting
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# load data
data_o, data_s, data_a, train_loader, val_loader, test_loader = load_data(args)


# train and test model
model, optimizer = Create_model(args)
train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args)

