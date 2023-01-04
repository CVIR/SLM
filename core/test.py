import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import params
from utils import make_variable, save_model
import copy

def eval_model(shared_model, data_loader):

    shared_model.eval()
    loss = 0.0
    acc = 0.0

    criterion = nn.CrossEntropyLoss().cuda()
    start_time = time.process_time()
    part = 0.0

    for step, (images, labels) in enumerate(data_loader):

        images = make_variable(images, gpu_id=params.src_gpu_id, volatile=True)
        labels = make_variable(labels, gpu_id=params.src_gpu_id)

        _, preds = shared_model(images,  [1])

        loss += criterion(preds, labels).data.item()

        _, preds_val = torch.max(preds.data, 1)
        acc += torch.sum(preds_val == labels.data)

        del images, labels, preds_val, preds

    avg_loss = loss / len(data_loader)
    part_avg = part / len(data_loader)

    avg_acc = float(acc.cpu().numpy()) / len(data_loader.dataset)

    end_time = time.process_time()
    print("Testing ended in {} secs.".format(end_time - start_time))
    print("Avg Loss = {}, Avg Acc = {}".format(avg_loss, str(avg_acc)))

    return avg_loss, avg_acc
