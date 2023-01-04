#SLM Framework training code
import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
import numpy as np
import params
from utils import *
from models.resnet50 import *
import time
import datetime
import copy
from torch.autograd import Function

import warnings
warnings.filterwarnings("ignore")


class TripletLoss(nn.Module):
    def __init__(self, margin=100.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.avg_hd = AveragedHausdorffLoss()
            
    def forward(self, source_feat, target_feat, activation_rates):
        """
        Triplet Loss function
        
        """
        distance_positive = self.avg_hd.forward(source_feat, target_feat, activation_rates)
        distance_negative = self.avg_hd.forward(source_feat, target_feat, 1.0 - activation_rates)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2, activation_rates):
        """
        Average Hausdorff Loss computation

        """
        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = torch.cdist(set1, set2) * activation_rates

        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])
        res = term_1 + term_2

        return res

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        """
        Gradient Reversal Layer (GRL)
        
        """
        return x

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def print_line():
    print('-'*100)

def get_lambda_discriminator(x):
    if x>= 1.0:
        return 1.0
    den = 1.0 + math.exp(-10 * x)
    lamb = (2.0 / den) - 1.0
    return lamb

def entropy(feat, lamda=1.0):
    # Obtain entropy loss value    
    feat = F.softmax(feat, dim=0)
    feat = feat.clamp(1e-15, 1-1e-15)
    loss_ent = -lamda * torch.mean(torch.sum(feat * (torch.log(feat)), 0))
    return loss_ent

def entropy_weight(feat, lamda=1.0):
    # Entropy contioning weights    
    feat = F.softmax(feat, dim=1)
    feat = feat.clamp(1e-15, 1-1e-15)
    loss_ent = -lamda * (torch.sum(feat * (torch.log(feat)), 1))
    return loss_ent

def mixup_data(x, y, alpha=2.0, use_cuda=True):
    # Returns mixed inputs, pairs of targets, and lambda' -- For Intra-domain MixUp
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam, index

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # Cross Entropy loss after smoothing the labels
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)

        return loss

def Entropy(input_):
    # Obtain entropy loss value
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def soft_cross_entropy(preds, target, reduction=None):
    # Cross entropy loss for soft labels
    log_probs = F.log_softmax(preds, dim=-1)
    return - (target * log_probs)

#####...Warmstart Model...#####
def warmstart_model(shared_model, src_data_loader, tgt_data_loader, data_loader_eval, num_iterations):

    writer = SummaryWriter(params.model_root)

    shared_model.train()

    discriminator_network = AdversarialNetwork(256)
    discriminator_network.cuda()
    discriminator_network.train()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD([ {"params": [param for name, param in shared_model.network.named_parameters() if 'fc' not in name], "lr": params.learning_rate * 0.1},
                                {"params": shared_model.network.fc.parameters(), "lr": params.learning_rate},
                                {"params": shared_model.last_layer.parameters(), "lr": params.learning_rate},
                                {"params": discriminator_network.parameters(), "lr": params.learning_rate}],
                              lr=params.learning_rate,
                              weight_decay=0.0005,
                              momentum=params.momentum)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(num_iterations))

    len_source_data_loader = len(src_data_loader) - 1 
    len_target_data_loader = len(tgt_data_loader) - 1

    print_line()
    print('len_source_data_loader = '+ str(len_source_data_loader))
    print('len_target_data_loader = '+ str(len_target_data_loader))

    best_accuracy_yet = 0.0
    best_itrn = 0
    best_model_wts = copy.deepcopy(shared_model.state_dict())

    for p in shared_model.network.parameters():
        p.requires_grad = False
    for p in shared_model.network.layer4.parameters():
        p.requires_grad = True
    for p in shared_model.network.fc.parameters():
        p.requires_grad = True

    print_line()
    print_line()
    print_line()

    start_time = time.process_time()
    epoch_number = 0

    for itrn in range(num_iterations):

        if itrn % len_source_data_loader == 0:
            iter_source = iter(src_data_loader)
            epoch_number = epoch_number + 1
        if itrn % len_target_data_loader == 0:
            iter_target = iter(tgt_data_loader)

        images_src, labels = iter_source.next()
        images_tgt, _ = iter_target.next()

        images_src = make_variable(images_src, gpu_id=params.src_gpu_id)
        images_tgt = make_variable(images_tgt, gpu_id=params.tgt_gpu_id)
        labels = make_variable(labels)

        optimizer.zero_grad()

        preds_src_adapt, preds_src = shared_model(images_src, [0])
        preds_tgt_adapt, preds_tgt = shared_model(images_tgt, [1])

        adv_label_source = torch.zeros_like(labels).cuda().long() # source --> 0
        adv_label_target = torch.ones_like(labels).cuda().long() # target --> 1

        adapt_loss = torch.tensor(0.0).cuda()

        lambd_discrim = get_lambda_discriminator(itrn/num_iterations)

        source_discrim_logits = discriminator_network(preds_src_adapt, lambd_discrim)
        source_adv_loss = (F.cross_entropy(source_discrim_logits, adv_label_source, reduction='none')).mean()

        target_discrim_logits = discriminator_network(preds_tgt_adapt, lambd_discrim)
        target_adv_loss = (F.cross_entropy(target_discrim_logits, adv_label_target, reduction='none')).mean()

        adapt_loss = source_adv_loss + target_adv_loss

        cls_loss = (F.cross_entropy(preds_src, labels, reduction='none')).mean()

        loss = cls_loss + adapt_loss
        loss.backward()
       
        optimizer.step()        
        scheduler.step()

        if ((itrn + 1) % params.log_step_freq == 0):
            print_line()
            print('Itrn:', itrn+1, 'LR:', scheduler.get_lr())
            print("Iteration [{}/{}]: cls_loss={} adapt_loss={} total_loss={}"
                  .format(itrn + 1,
                          num_iterations,
                          cls_loss.item(),
                          adapt_loss.item(),
                          loss.item()))
            writer.add_scalar('Train_Loss/train_itrn', loss.item(), itrn)
            writer.add_scalar('Cls_Loss/train_itrn', cls_loss.item(), itrn)
            writer.add_scalar('Adapt_Loss/train_itrn', adapt_loss.item(), itrn)
            print_line()
            print_line()
            end_time = time.process_time()
            print('Time taken:: {} secs'.format(end_time-start_time))
            print_line()
            start_time = time.process_time()

        if ((itrn + 1) % params.eval_step_freq == 0):
            loss_val = 0.0
            acc_val = 0.0
            start_time_eval = time.process_time()
            tot_len = 0.0
            shared_model.eval()
            for step_val, (images_val, labels_val) in enumerate(data_loader_eval):
                if step_val % 100 == 0:
                    print("\rEvaluating batch {}/{}".format(step_val, len(data_loader_eval)), end='', flush=True)

                images_val = make_variable(images_val, gpu_id=params.src_gpu_id, volatile=True)
                labels_val = make_variable(labels_val, gpu_id=params.src_gpu_id)

                tot_len += len(labels_val)

                _, preds_val = shared_model(images_val, [1])

                loss_val += criterion(preds_val, labels_val).data.item()

                _, preds_val_val = torch.max(preds_val.data, 1)
                acc_val += torch.sum(preds_val_val == labels_val.data)

                del images_val, labels_val, preds_val, preds_val_val

            print('acc_val: ', acc_val)
            avg_loss_val = loss_val / len(data_loader_eval)
            avg_acc_val = float(acc_val.cpu().numpy()) / len(data_loader_eval.dataset)

            if(best_accuracy_yet <= avg_acc_val):
                best_accuracy_yet = avg_acc_val
                best_model_wts = copy.deepcopy(shared_model.state_dict())
                best_itrn = itrn + 1
            print('best_acc_yet: ', best_accuracy_yet, ' ( in itrn:', best_itrn, ' )...')

            shared_model.train()
            print_line()

            end_time_eval = time.process_time()
            print(" -- Evaluation ended in {} secs.".format(end_time_eval - start_time_eval))
            print("Avg Loss = {}, Avg Acc = {}".format(avg_loss_val, str(avg_acc_val)))
            writer.add_scalar('Test_Loss/train_itrn', avg_loss_val, itrn + 1)
            writer.add_scalar('Test_Acc/train_itrn', avg_acc_val, itrn + 1)

            print_line()

        del images_src, images_tgt, labels, preds_src, preds_tgt

    print('Loading the best model weights...')
    shared_model.load_state_dict(best_model_wts)
    save_model(shared_model, "Shared-Model-Best-WS-{}.pth".format(best_itrn))
    print_line()
    print_line()

    return shared_model

#####...Train SLM...#####
def train_slm(shared_model, src_data_loader, tgt_data_loader, data_loader_eval, num_iterations):
    # Trainer function
    
    writer = SummaryWriter(params.model_root)
    
    shared_model.train()

    selector_network = Selector_Network()
    selector_network.cuda()
    selector_network.train()

    discriminator_network = AdversarialNetwork(in_feature=256)
    discriminator_network.cuda()
    discriminator_network.train()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD([ {"params": [param for name, param in shared_model.network.named_parameters() if 'fc' not in name], "lr": params.learning_rate * 0.1},
                            {"params": shared_model.network.fc.parameters(), "lr": params.learning_rate},
                            {"params": shared_model.last_layer.parameters(), "lr": params.learning_rate},
                            {"params": discriminator_network.parameters(), "lr": params.learning_rate}],
                            lr=params.learning_rate,
                            weight_decay=0.0005,
                            momentum=params.momentum )
    optimizer_selector = optim.SGD(selector_network.parameters(),
                                    lr=10.0*params.learning_rate,
                                    weight_decay=0.001,
                                    momentum=params.momentum)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(num_iterations))
    scheduler_selector = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_selector, float(num_iterations))

    len_source_data_loader = len(src_data_loader) - 1 
    len_target_data_loader = len(tgt_data_loader) - 1

    print_line()
    print('len_source_data_loader = '+ str(len_source_data_loader))
    print('len_target_data_loader = '+ str(len_target_data_loader))

    best_accuracy_yet = 0.0
    best_itrn = 0
    best_model_wts = copy.deepcopy(shared_model.state_dict())
    best_discrim_wts = copy.deepcopy(discriminator_network.state_dict())
    best_selector_wts = copy.deepcopy(selector_network.state_dict())

    for p in shared_model.network.parameters():
        p.requires_grad = False
    for p in shared_model.network.layer4.parameters():
        p.requires_grad = True
    for p in shared_model.network.fc.parameters():
        p.requires_grad = True

    print_line()
    print_line()
    print_line()

    start_time = time.process_time()
    running_lr = params.learning_rate
    epoch_number = 0

    # if params.warmstart_models=='True':
    #     print('Warmstarting...')
    #     shared_model.load_state_dict(torch.load(params.warmstart_model_from))
    #     print('Warmstarted successfully...')

    set2set_margin_loss = TripletLoss()
    for itrn in range(num_iterations):

        if itrn%100 == 0:
            print('Itrn: (T)', itrn+1, 'LR:', scheduler.get_lr(), ' selector_lr: ', scheduler_selector.get_lr())
        if itrn % len_source_data_loader == 0:
            iter_source = iter(src_data_loader)
            epoch_number = epoch_number + 1
        if itrn % len_target_data_loader == 0:
            iter_target = iter(tgt_data_loader)

        images_src, labels = iter_source.next()
        images_tgt, _ = iter_target.next()

        images_src = make_variable(images_src, gpu_id=params.src_gpu_id)
        images_tgt = make_variable(images_tgt, gpu_id=params.tgt_gpu_id)
        labels = make_variable(labels)

        mix_ratio = np.random.beta(2.0, 2.0)
        mix_ratio = round(mix_ratio, 2)
        
        clip_thr = 0.1
        # clip the mixup_ratio
        if (mix_ratio >= 0.5 and mix_ratio < (0.5 + clip_thr)):
            mix_ratio = 0.5 + clip_thr
        if (mix_ratio > (0.5 - clip_thr) and mix_ratio < 0.5):
            mix_ratio = 0.5 - clip_thr

        mix_label = torch.FloatTensor(params.batch_size).fill_(mix_ratio)
        mix_label = make_variable(mix_label)

        images_mix = (mix_ratio * images_src) + ((1 - mix_ratio) * images_tgt)

        optimizer.zero_grad()
        optimizer_selector.zero_grad()

        preds_src_adapt, preds_src = shared_model(images_src, [0])
        preds_tgt_adapt, preds_tgt = shared_model(images_tgt, [1])
        preds_mix_adapt_src, preds_mix_src = shared_model(images_mix, [0])
        preds_mix_adapt_tgt, preds_mix_tgt = shared_model(images_mix, [1])

        lambd_discrim = get_lambda_discriminator(itrn/num_iterations) #

        activation_logits = selector_network(images_src)
        activation_hard = F.gumbel_softmax(activation_logits, tau=1-lambd_discrim+1e-5, hard=True)

        act_loss = entropy(activation_logits[:,1], lamda=10.0)

        activation_rates = activation_hard[:,1]
        activation_rates = activation_rates.squeeze()

        adv_label_source = torch.zeros_like(labels).cuda().long() # 0 for source
        adv_label_target = torch.ones_like(labels).cuda().long() # 1 for target


        part = torch.mean(activation_rates)

        adapt_loss = torch.tensor(0.0).cuda()

        # Supervision loss from labeled source samples.
        cls_loss = (CrossEntropyLabelSmooth(num_classes=31, epsilon=0.2, size_average=False)(preds_src, labels) * activation_rates.detach()).mean()
        

        source_discrim_logits = discriminator_network(preds_src_adapt, lambd_discrim)
        source_ent_weights = entropy_weight(preds_src)
        source_ent_weights = source_ent_weights.squeeze().detach()
        source_adv_loss = (F.cross_entropy(source_discrim_logits, adv_label_source, reduction='none') * source_ent_weights * GradReverse(lambd_discrim)(activation_rates)).mean()

        target_discrim_logits = discriminator_network(preds_tgt_adapt, lambd_discrim)
        target_ent_weights = entropy_weight(preds_tgt)
        target_ent_weights = target_ent_weights.squeeze().detach()
        target_adv_loss = (F.cross_entropy(target_discrim_logits, adv_label_target, reduction='none') * target_ent_weights).mean()

        mix_source_discrim_logits = discriminator_network(preds_mix_adapt_src, lambd_discrim)
        mix_source_adv_loss = (F.cross_entropy(mix_source_discrim_logits, adv_label_source, reduction='none') * mix_label * GradReverse(lambd_discrim)(activation_rates)).mean()

        mix_target_discrim_logits = discriminator_network(preds_mix_adapt_tgt, lambd_discrim)
        mix_target_adv_loss = (F.cross_entropy(mix_target_discrim_logits, adv_label_target, reduction='none') * (1.0 - mix_label) * GradReverse(lambd_discrim)(activation_rates)).mean() # Added the source selection part

        # Domain Discriminator loss.
        adapt_loss = source_adv_loss + target_adv_loss + mix_source_adv_loss + mix_target_adv_loss

        target_logits_softmax = torch.softmax(preds_tgt.data, dim=-1)
        max_probs, target_pseudo_labels = torch.max(target_logits_softmax, dim=-1)
        pseudo_mask = max_probs.ge(params.pseudo_threshold).float()

        target_soft_pseudolabels = torch.softmax(preds_tgt.data/(0.05*(1-lambd_discrim+1e-5)), dim=-1)
        
        #Regulariser for "Select" module.
        softmax_out = nn.Softmax(dim=1)(preds_tgt)
        entropy_loss = torch.mean(Entropy(softmax_out))        
        msoftmax = softmax_out.mean(dim=0)
        entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

        reg_sel_loss = 0.1 * entropy_loss

        pseudo_cls_loss =  (soft_cross_entropy(F.softmax(preds_tgt), target_soft_pseudolabels, reduction='none').sum(1) * pseudo_mask).mean()

        # Intra-Domain adaptation.
        intra_source_mixed, label_i_src_a, label_i_src_b, lam_src, index_src = mixup_data(images_src, labels)
        intra_target_mixed, label_i_tgt_a, label_i_tgt_b, lam_tgt, index_tgt = mixup_data(images_tgt, target_soft_pseudolabels)

        preds_Intra_mix_adapt_src, preds_Intra_mix_src = shared_model(intra_source_mixed, [0])
        preds_Intra_mix_adapt_tgt, preds_Intra_mix_tgt = shared_model(intra_target_mixed, [1])

        intra_mix_cls_loss_src = (CrossEntropyLabelSmooth(num_classes=31, epsilon=0.2, size_average=False)(preds_Intra_mix_src, label_i_src_a) * lam_src * activation_rates.detach() * activation_rates[index_src].detach()).mean() +\
                                    (CrossEntropyLabelSmooth(num_classes=31, epsilon=0.2, size_average=False)(preds_Intra_mix_src, label_i_src_b) * (1.0 - lam_src) * activation_rates.detach() * activation_rates[index_src].detach()).mean()
        intra_mix_cls_loss_tgt = (soft_cross_entropy(F.softmax(preds_Intra_mix_tgt), (label_i_tgt_a*lam_tgt) + label_i_tgt_b*(1.0-lam_tgt), reduction='none').sum(1) * pseudo_mask * pseudo_mask[index_tgt]).mean()

        intra_mixup_cls_loss = intra_mix_cls_loss_src + intra_mix_cls_loss_tgt

        intra_mix_source_discrim_logits = discriminator_network(preds_Intra_mix_adapt_src, lambd_discrim)
        intra_mix_source_adv_loss = (F.cross_entropy(intra_mix_source_discrim_logits, adv_label_source, reduction='none') * mix_label * GradReverse(lambd_discrim)(activation_rates * activation_rates[index_src])).mean()

        intra_mix_target_discrim_logits = discriminator_network(preds_Intra_mix_adapt_tgt, lambd_discrim)
        intra_mix_target_adv_loss = (F.cross_entropy(intra_mix_target_discrim_logits, adv_label_target, reduction='none') * (1.0 - mix_label)).mean()

        intra_mixup_dom_loss = intra_mix_source_adv_loss + intra_mix_target_adv_loss
        all_intra_mix_loss = intra_mixup_cls_loss + intra_mixup_dom_loss

        mix_cls_loss_src = (CrossEntropyLabelSmooth(num_classes=31, epsilon=0.2, size_average=False)(preds_mix_src, labels) * mix_label * activation_rates.detach()).mean()
        mix_cls_loss_tgt = (soft_cross_entropy(F.softmax(preds_mix_tgt), target_soft_pseudolabels * (1.0 - mix_ratio), reduction='none').sum(1) * activation_rates.detach() * pseudo_mask).mean()

        mix_loss = mix_cls_loss_src + mix_cls_loss_tgt + all_intra_mix_loss

        retro_loss = torch.tensor(0.0).cuda()

        # Hausdorff distance loss.
        set2set_loss = 0.01 * set2set_margin_loss.forward(preds_src_adapt.detach(), preds_tgt_adapt.detach(), activation_rates)

        loss = set2set_loss + cls_loss + pseudo_cls_loss + adapt_loss + mix_loss + act_loss + reg_sel_loss
        loss.backward()
     
        optimizer_selector.step()
        optimizer.step()
        
        scheduler.step()
        scheduler_selector.step()

        # Log updates.
        if ((itrn + 1) % params.log_step_freq == 0):
            print_line()
            print("Iteration (T) [{}/{}]: cls_loss={} adapt_loss={} slm_loss={}"
                  .format(itrn + 1,
                          num_iterations,
                          cls_loss.item(),
                          adapt_loss.item(),
                          set2set_loss.item() + reg_sel_loss.item() + act_loss.item() + pseudo_cls_loss.item() + mix_loss.item()
                          ))
            writer.add_scalar('Train_Loss/train_itrn', loss.item(), itrn)
            writer.add_scalar('Cls_Loss/train_itrn', cls_loss.item(), itrn)
            writer.add_scalar('Adapt_Loss/train_itrn', adapt_loss.item(), itrn)
            print_line()
            print_line()
            end_time = time.process_time()
            print('Time taken:: {} secs'.format(end_time-start_time))
            print_line()
            start_time = time.process_time()

        # Evaluate model on the validation set.
        if ((itrn + 1) % params.eval_step_freq == 0):
            loss_val = 0.0
            acc_val = 0.0
            start_time_eval = time.process_time()
            tot_len = 0.0
            shared_model.eval()
            for step_val, (images_val, labels_val) in enumerate(data_loader_eval):
                if step_val % 100 == 0:
                    print("\rEvaluating batch {}/{}".format(step_val, len(data_loader_eval)), end='', flush=True)

                images_val = make_variable(
                    images_val, gpu_id=params.src_gpu_id, volatile=True)
                labels_val = make_variable(
                    labels_val, gpu_id=params.src_gpu_id)

                tot_len += len(labels_val)

                _, preds_val = shared_model(images_val, [1])

                loss_val += criterion(preds_val, labels_val).data.item()

                _, preds_val_val = torch.max(preds_val.data, 1)
                acc_val += torch.sum(preds_val_val == labels_val.data)

                del images_val, labels_val, preds_val, preds_val_val

            print('acc_val: ', acc_val)

            avg_loss_val = loss_val / len(data_loader_eval)
            avg_acc_val = float(acc_val.cpu().numpy()) / len(data_loader_eval.dataset)

            if(best_accuracy_yet <= avg_acc_val):
                best_accuracy_yet = avg_acc_val
                best_model_wts = copy.deepcopy(shared_model.state_dict())
                best_selector_wts = copy.deepcopy(selector_network.state_dict())
                best_discrim_wts = copy.deepcopy(discriminator_network.state_dict())
                best_temperature = 1-lambd_discrim+1e-5
                best_itrn = itrn + 1

            print('best_acc_yet: ', best_accuracy_yet, ' ( in itrn:', best_itrn, ')...')
            shared_model.train()
            print_line()

            end_time_eval = time.process_time()
            print("Avg Loss = {}, Avg Acc = {}".format(avg_loss_val, str(avg_acc_val)))
            writer.add_scalar('Test_Loss/train_itrn', avg_loss_val, itrn + 1)
            writer.add_scalar('Test_Acc/train_itrn', avg_acc_val, itrn + 1)
            print_line()

        # Save Model parameters.
        if ((itrn + 1) % params.save_step_freq == 0):
            save_model(shared_model, "Shared-Model-{}.pth".format(itrn + 1))
            save_model(selector_network, "Shared-Selector-{}.pth".format(itrn + 1))
            save_model(discriminator_network, "Shared-Discriminator-{}.pth".format(itrn + 1))

        del images_src, images_tgt, labels, preds_src, preds_tgt, activation_rates

    # Save the model in the final iteration.
    save_model(shared_model, "Shared-Model-{}.pth".format(itrn + 1))
    save_model(selector_network, "Shared-Selector-{}.pth".format(itrn + 1))
    save_model(discriminator_network, "Shared-Discriminator-{}.pth".format(itrn + 1))

    writer.close()

    # Load the best models and save them.
    print('Loading the best model weights...')
    shared_model.load_state_dict(best_model_wts)
    selector_network.load_state_dict(best_selector_wts)
    discriminator_network.load_state_dict(best_discrim_wts)
    save_model(shared_model, "Shared-Model-Best-{}.pth".format(best_itrn))
    save_model(selector_network, "Selector-Model-Best-{}.pth".format(best_itrn))
    save_model(discriminator_network, "Discriminator-Model-Best-{}.pth".format(best_itrn))    

    print_line()
    print_line()
    print('SLM Training Completed.')
    print('Accuracy Obtained: ', best_accuracy_yet)
    print_line()
    print_line()

    return shared_model
