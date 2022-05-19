import numpy as np
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

import random

from sklearn.metrics import confusion_matrix

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def get_acc(y_pred, y_true):
    """ ACC metric
    y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
    y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
    task: the task of the current dataset(multi-label or multi-class)
    threshold: the threshold for multilabel
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    y_pred = np.argmax(y_pred, axis=1)
    correct = np.sum(np.equal(y_true, y_pred))
    total = y_true.shape[0]
    
    return correct, total

def get_wp_f1(y_pred, y_true):
    """ Precision_Recall_F1score metrics
    y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
    y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
    """
    eps=1e-20
    y_pred = torch.argmax(y_pred, dim=1)

    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    # F1_sci = f1_score(y_true, y_pred, average=None)
    confusion = confusion_matrix(y_true, y_pred)

    f1_dict = {}
    precision_list = []
    TP_list = []
    FN_list = []
    for i in range(len(confusion)):
        TP = confusion[i, i]
        FP = sum(confusion[:, i]) - TP
        FN = sum(confusion[i, :]) - TP

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        result_f1 = 2 * precision  * recall / (precision + recall + eps)

        TP_list.append(TP)
        FN_list.append(FN)
        f1_dict[i] = result_f1
        precision_list.append(precision)

    total_image = y_pred.shape[0]
    weighted = 0.
    for i in range(len(confusion)):
        weighted += precision_list[i] * (TP_list[i] + FN_list[i])

    WP = weighted / total_image

    return f1_dict, WP