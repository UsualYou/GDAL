import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# Custom
from config import *
from data.sampler import SubsetSequentialSampler


def get_uncertainty(models, unlabeled_loader, args=None):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        sub_logit_all = torch.tensor([])
        pred_label_all = torch.tensor([])

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            main_logit, _, features = models['backbone'](inputs)

            _, pred_label = torch.max(main_logit, dim=1)
            pred_label = pred_label.detach().cpu()
            pred_label_all = torch.cat((pred_label_all, pred_label), 0)

            sub_logit = models['module'](features)
            sub_logit = sub_logit.detach().cpu()
            sub_logit_all = torch.cat((sub_logit_all, sub_logit), 0)

    sub_prob = torch.softmax(sub_logit_all, dim=1)

    # get uncertainty
    sub_entropy = -(sub_prob * torch.log(sub_prob)).sum(dim=1)  # Entropy
    pred_prob = sub_prob[torch.arange(sub_prob.size(0)), pred_label_all.long()]
    if sub_prob.size(0) > 1:
        pred_prob_var = torch.var(pred_prob)
        mean_pred_prob = torch.mean(pred_prob)
        direction_diversity = torch.abs(pred_prob - mean_pred_prob) / (torch.sqrt(pred_prob_var) + 1e-8)
    else:
        direction_diversity = torch.ones_like(sub_entropy)
    uncertainty = sub_entropy * direction_diversity

    return uncertainty.cpu()


# Select the indices of the unlablled data according to the methods
def query_samples(model, data_unlabeled, subset, args):
    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                  sampler=SubsetSequentialSampler(subset),
                                  pin_memory=True, num_workers=args.num_workers)

    uncertainty = get_uncertainty(model, unlabeled_loader, args)
    arg = np.argsort(uncertainty)

    return arg
