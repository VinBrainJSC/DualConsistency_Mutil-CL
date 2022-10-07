"""
Training code for MTCL(c)
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time
import json
from skimage import measure
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets_IRCAD_SSL_concat_kfold, RandomGenerator_IRCAD_concat,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_unet_2D import test_single_concat_volume
from medpy import metric
from scipy.ndimage import zoom
# HD loss and boundary loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

# Confident Learning module
import cleanlab
from albumentations import *

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0
    
    
def post_processing(prediction):
    label_cc, num_cc = measure.label(prediction, return_num=True)
    total_cc = np.sum(prediction)
    for cc in range(1, num_cc+1):
        single_cc = (label_cc == cc)
        single_vol = np.sum(single_cc)
        # remove small regions
        if single_vol/total_cc < 0.001:
            prediction[single_cc] = 0
    return prediction


limit=20
interpolation=1
border_mode=4
value=None
mask_value=None

rotate = Rotate(limit, interpolation, border_mode, value, mask_value, p=1.0)
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/IRCAD_c', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='IRCAD_c/CPS_MCL_c_Sato_fix_rotate_Unet_urpc_MCL_ICT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[2, 320, 320],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1000, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')
parser.add_argument('--fold', type=int, default=2,
                    help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# pretrain
parser.add_argument('--pretrain_model', type=str, default='/vinbrain/tuanvv/MTCL-main/model/IRCAD_c/CPS_MCL_c_Sato_fix_rotate_Unet_urpc_ICT_verify_full_LQCPS_labeled/unet_urpc_kfold/fold_5/unet_urpc_best_model_dice_0.7069.pth', help='pretrained model')

# CL
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
args = parser.parse_args()


# BD and HD loss
def compute_dtm01(img_gt, out_shape):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """

    normalized_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis / np.max(posdis)

    return normalized_dtm


def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)

    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                            np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf

    return normalized_sdf


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary == 1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf


def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:, 1, ...]
    dc = gt_sdf[:, 1, ...]
    multipled = torch.mul(pc, dc)
    bd_loss = multipled.mean()

    return bd_loss


def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:, 1, ...] - gt.float()) ** 2  # [4, 320, 320]
    # print('delta shape:', delta_s.shape)
    s_dtm = seg_dtm[:, 1, ...] ** 2
    g_dtm = gt_dtm[:, 1, ...] ** 2
    dtm = s_dtm + g_dtm  # [4, 320, 320]
    # print('dtm shape:', dtm.shape)
    multipled = torch.mul(delta_s, dtm)
    # print('dtm shape:', multipled.shape)
    hd_loss = multipled.mean()

    return hd_loss


def labeled_slices(dataset, fold_num):
#     ref_dict = None
    if "IRCAD" in dataset:  # 1-1298 are IRCAD slices, others are MSD slices
        with open(dataset + '/msd_ir_folds.json', 'r') as f1:
            sample_list = json.load(f1)
            sample_list = sample_list['fold_{}'.format(fold_num)]['train_fold_{}'.format(fold_num)]
    else:
        raise Exception("Error")
    
    # Return the number of training samples in IRCAD
    count = 0
    for i in range(len(sample_list)):
        if 'zimage' in sample_list[i]:
            print(sample_list[i])
            break
        else:
            count += 1
#     print(count)
#     print(sample_list[count])
    return count


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path,fold_num):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    pretrain_model = args.pretrain_model
    # print('asdadsada',fold_num)
    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=2,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # using pretrain?
    if True:
        model.load_state_dict(torch.load(pretrain_model))
        print("Loaded Pretrained Model")
#     ema_model = create_model(ema=True)
    
    print('Here')
    

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    db_val = BaseDataSets_IRCAD_SSL_concat_kfold(base_dir=args.root_path, split="val",fold_num=fold_num)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
#     writer = SummaryWriter(snapshot_path + '/log/fold_{}'.format(fold_num))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        metric_i = test_single_concat_volume(
            sampled_batch["image"], sampled_batch["label_ROI"], model, classes=num_classes)
        metric_list += np.array(metric_i)
#         break
    metric_list = metric_list / len(db_val)
#     for class_i in range(num_classes - 1):
#         writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
#                           metric_list[class_i, 0], iter_num)
#         writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
#                           metric_list[class_i, 1], iter_num)

    performance = np.mean(metric_list, axis=0)[0]

    mean_hd95 = np.mean(metric_list, axis=0)[1]
#     writer.add_scalar('info/val_mean_dice', performance, iter_num)
#     writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)


    logging.info(
        'mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
       

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_labeled/{}_kfold".format(
        args.exp, args.model)
    print(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    for fold_num in range(1,6):
        if fold_num != args.fold:
            continue
        print(fold_num)
        logging.basicConfig(filename=snapshot_path + "/log_fold_{}.txt".format(fold_num), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        train(args, snapshot_path,fold_num)
