import gc
import os
import wandb
import pytz
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libriphone import LibriDataset, preprocess_data
from network import Classifier, RegLSTM

import numpy as np
from tqdm import tqdm
from lstm import LstmParam, LstmNetwork

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=41, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

# # data prarameters
# concat_nframes = 21  # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
# train_ratio = 0.8  # the ratio of data used for training, the rest will be used for validation
#
# # training parameters
# seed = 0  # random seed
# batch_size = 2048  # batch size
# num_epoch = 50  # the number of training epoch
# early_stopping = 8
# learning_rate = 0.001  # learning rate
# model_path = './model.ckpt'  # the path where the checkpoint will be saved
#
# # model parameters
# input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
# hidden_layers = 5  # the number of hidden layers
# hidden_dim = 1024  # the hidden dim

# data prarameters
cfg = {"train_ratio": 0.95,
       "seed": 0,
       "num_epoch": 30,
       "early_stopping": 8,
       "batch_size": 2048,
       "learning_rate": 0.0001,
       "weight_decay": 0.01,
       "dropout": 0.3,
       'grad_norm_max': 10,
       "bidirectional": True,
       "exp_name": "ML_HW2_strong_baseline",
       "model_path": "./output/model.ckpt",
       "concat_nframes": 21,
       "input_dim": 39,  # * concat_nframes,
       "hidden_layers": 3,
       "hidden_dim": 128
       }

if torch.cuda.is_available():
    device = 'cuda'
    print("GPU is available. Using CUDA.")
else:
    print("GPU is not available. Using CPU.")
    device = 'cpu'

if not os.path.exists("./output"):
    os.mkdir("./output")

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                   concat_nframes=cfg["concat_nframes"], train_ratio=cfg["train_ratio"])
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone',
                               concat_nframes=cfg["concat_nframes"], train_ratio=cfg["train_ratio"])

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)


# fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# fix random seed
same_seeds(cfg["seed"])

wandb.login()  # 用正确key
cn_tz = pytz.timezone('Asia/Shanghai')
now_time = datetime.datetime.now(cn_tz).strftime('%Y-%m-%d %H:%M:%S')

run = wandb.init(project=cfg['exp_name'], config=cfg, name=now_time, save_code=True)

# create model, define a loss function, and optimizer
# model = Classifier(input_dim=cfg["input_dim"], hidden_layers=cfg["hidden_layer"], hidden_dim=cfg["hidden_dim"]).to(device)
model = RegLSTM(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["hidden_layers"],
                dropout=cfg["dropout"], bidirectional=cfg["bidirectional"]).to(device)
# criterion = nn.CrossEntropyLoss()
criterion = CrossEntropyLabelSmooth()

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
# schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=learning_rate/10)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epoch"],
                                                      eta_min=cfg["learning_rate"] / 100)

best_acc = 0.0
early_stop_count = 0
for epoch in range(cfg["num_epoch"]):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    lr = optimizer.param_groups[0]["lr"]
    # training
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['grad_norm_max'])
        optimizer.step()

        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()

    schedule.step()

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_set)

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                val_acc += (
                        val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_acc / len(val_set)
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, cfg["num_epoch"], train_acc, train_loss, val_acc, val_loss
            ))

            run.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
                     "val_acc": val_acc, "now_lr": lr})

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), cfg["model_path"])
                print('saving model with acc {:.3f}'.format(best_acc))
                run.log({"best_epoch": epoch + 1, "best_acc": best_acc})
            else:
                early_stop_count += 1
                if early_stop_count >= cfg["early_stopping"]:
                    print(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                    break
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, cfg["num_epoch"], train_acc, train_loss
        ))
        run.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "now_lr": lr})

run.finish()

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), cfg["model_path"])
    print('saving model at last epoch')

del train_loader, val_loader
gc.collect()

# load data
test_X = preprocess_data(split='test', feat_dir='/kaggle/input/ml2022spring-hw2/libriphone/libriphone/feat',
                         phone_path='/kaggle/input/ml2022spring-hw2/libriphone/libriphone',
                         concat_nframes=cfg["concat_nframes"])
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

# load model
# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model = RegLSTM(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["hidden_layers"],
                dropout=cfg["dropout"], bidirectional=cfg["bidirectional"]).to(device)
model.load_state_dict(torch.load(cfg["model_path"]))

test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

with open('./output/prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))
