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
from val_unet_2D import test_single_concat_volume, calculate_metric_percase

# HD loss and boundary loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

# Confident Learning module
import cleanlab

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/IRCAD_c', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='IRCAD_c/CPS_c_Sato_fix_rotate_Unet_urpc_MCL_ICT', help='experiment_name')
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
parser.add_argument('--gpu', type=str, default='1',
                    help='gpu id')

parser.add_argument('--fold', type = int, default = None,
                   help = 'fold number')

# label and unlabel
parser.add_argument('--ict_alpha', type=int, default=0.2,
                    help='ict_alpha')
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
parser.add_argument('--pretrain_model', type=str, default=None, help='pretrained model')

# CL
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
args = parser.parse_args()

#swin
# from networks.config import get_config
# parser.add_argument(
#     '--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
# parser.add_argument(
#     "--opts",
#     help="Modify config options by adding 'KEY VALUE' pairs. ",
#     default=None,
#     nargs='+',
# )
# parser.add_argument('--zip', action='store_true',
#                     help='use zipped dataset instead of folder dataset')
# parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                     help='no: no cache, '
#                     'full: cache all data, '
#                     'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
# parser.add_argument('--resume', help='resume from checkpoint')
# parser.add_argument('--accumulation-steps', type=int,
#                     help="gradient accumulation steps")
# parser.add_argument('--use-checkpoint', action='store_true',
#                     help="whether to use gradient checkpointing to save memory")
# parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
#                     help='mixed precision opt level, if O0, no amp is used')
# parser.add_argument('--tag', help='tag of experiment')
# parser.add_argument('--eval', action='store_true',
#                     help='Perform evaluation only')
# parser.add_argument('--throughput', action='store_true',
#                     help='Test throughput only')


# args = parser.parse_args()
# config = get_config(args)

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
#             print(sample_list[i])
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

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=2,
                            class_num=num_classes)
#         model = net_factory(net_type=args.model, in_chns=2,
#                             class_num=num_classes,config = config)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # using pretrain?
    if pretrain_model:
        model.load_state_dict(torch.load(pretrain_model))
        print("Loaded Pretrained Model")
    ema_model = create_model()
    
    print('Here')
    

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets_IRCAD_SSL_concat_kfold(base_dir=args.root_path, split="train", fold_num = fold_num, num=None, transform=transforms.Compose([
        RandomGenerator_IRCAD_concat(args.patch_size)]))

    db_val = BaseDataSets_IRCAD_SSL_concat_kfold(base_dir=args.root_path, split="val",fold_num = fold_num)

    total_slices = len(db_train)
    labeled_slice = labeled_slices(args.root_path, fold_num)
    print("Total slices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_ema = optim.SGD(ema_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    focal_loss = losses.FocalLoss()

    writer = SummaryWriter(snapshot_path + '/log/fold_{}'.format(fold_num))
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance_ema = 0.0
    kl_distance = nn.KLDivLoss(reduction='none')
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label_ROI']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:] ## the batch which was processed by teacher model
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            # create noise which value range from [-0.2,0.2] for teacher model's input
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)  
            ema_inputs = unlabeled_volume_batch + noise
            
            # Student model forward
            ### ict factors
            ict_mix_factors = np.random.beta(
                args.ict_alpha, args.ict_alpha, size=(args.labeled_bs//2, 1, 1, 1))
            ict_mix_factors = torch.tensor(
                ict_mix_factors, dtype=torch.float).cuda()
            unlabeled_volume_batch_0 = unlabeled_volume_batch[0:args.labeled_bs//2, ...]
            unlabeled_volume_batch_1 = unlabeled_volume_batch[args.labeled_bs//2:, ...]

            # Mix images
            batch_ux_mixed = unlabeled_volume_batch_0 * \
                (1.0 - ict_mix_factors) + \
                unlabeled_volume_batch_1 * ict_mix_factors
            input_volume_batch = torch.cat(
                [volume_batch, batch_ux_mixed], dim=0)
            
            outputs,outputs1,outputs2,outputs3 = model(input_volume_batch)
            
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs_soft3 = torch.softmax(outputs3, dim=1)
            
#             out = torch.argmax(outputs_soft, dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             out[out > 0 ] = 1
#             test = label_batch.cpu().detach().numpy()
#             test[test > 0] = 1
#             print(calculate_metric_percase(out == 1, test == 1))
            
#             with torch.no_grad():
            ema_output,ema_output1,ema_output2,ema_output3 = ema_model(volume_batch)
            ema_output_soft = torch.softmax(ema_output, dim=1)
            ema_output_soft1 = torch.softmax(ema_output1, dim=1)
            ema_output_soft2 = torch.softmax(ema_output2, dim=1)
            ema_output_soft3 = torch.softmax(ema_output3, dim=1)

            ema_output_i0,ema_output1_i0,ema_output2_i0,ema_output3_i0 = ema_model(unlabeled_volume_batch_0)
            ema_output_i1,ema_output1_i1,ema_output2_i1,ema_output3_i1 = ema_model(unlabeled_volume_batch_1)

            ema_output_soft_i0 = torch.softmax(ema_output_i0, dim=1)
            ema_output_soft1_i0 = torch.softmax(ema_output1_i0, dim=1)
            ema_output_soft2_i0 = torch.softmax(ema_output2_i0, dim=1)
            ema_output_soft3_i0 = torch.softmax(ema_output3_i0, dim=1)

            ema_output_soft_i1 = torch.softmax(ema_output_i1, dim=1)
            ema_output_soft1_i1 = torch.softmax(ema_output1_i1, dim=1)
            ema_output_soft2_i1 = torch.softmax(ema_output2_i1, dim=1)
            ema_output_soft3_i1 = torch.softmax(ema_output3_i1, dim=1)

            batch_pred_mixed = ema_output_soft_i0 * \
                (1.0 - ict_mix_factors) + ema_output_soft_i1 * ict_mix_factors
            batch_pred_mixed1 = ema_output_soft1_i0 * \
                (1.0 - ict_mix_factors) + ema_output_soft1_i1 * ict_mix_factors
            batch_pred_mixed2 = ema_output_soft2_i0 * \
                (1.0 - ict_mix_factors) + ema_output_soft2_i1 * ict_mix_factors
            batch_pred_mixed3 = ema_output_soft3_i0 * \
                (1.0 - ict_mix_factors) + ema_output_soft3_i1 * ict_mix_factors
            # Loss Supervised of HQ dataset : 2 first samples of batch 
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            loss_ce1 = ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice1 = dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_ce2 = ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice2 = dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            loss_ce3 = ce_loss(outputs3[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice3 = dice_loss(outputs_soft3[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            #### ema
            
            loss_ce_ema = ce_loss(ema_output[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice_ema = dice_loss(ema_output[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            loss_ce1_ema = ce_loss(ema_output1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice1_ema = dice_loss(ema_output1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_ce2_ema = ce_loss(ema_output2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice2_ema = dice_loss(ema_output2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            loss_ce3_ema = ce_loss(ema_output3[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice3_ema = dice_loss(ema_output3[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            supervised_loss_1 = (loss_ce + loss_dice + loss_ce1 + loss_dice1 + loss_ce2 + loss_dice2
                              + loss_ce3 + loss_dice3) / 8
            
            supervised_loss_ema = (loss_ce_ema + loss_dice_ema + loss_ce1_ema + loss_dice1_ema + loss_ce2_ema + loss_dice2_ema
                              + loss_ce3_ema + loss_dice3_ema) / 8
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            
            pseudo_outputs1_0 = torch.argmax(outputs_soft[args.labeled_bs:-1].detach(), dim=1, keepdim=False)
            pseudo_outputsema_0 = torch.argmax(ema_output[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1_0 = ce_loss(outputs[args.labeled_bs:-1], pseudo_outputsema_0)
            pseudo_supervision2_0 = ce_loss(ema_output[args.labeled_bs:], pseudo_outputs1_0)
            
            pseudo_outputs1_1 = torch.argmax(outputs_soft1[args.labeled_bs:-1].detach(), dim=1, keepdim=False)
            pseudo_outputsema_1 = torch.argmax(ema_output1[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1_1 = ce_loss(outputs1[args.labeled_bs:-1], pseudo_outputsema_1)
            pseudo_supervision2_1 = ce_loss(ema_output1[args.labeled_bs:], pseudo_outputs1_1)
            
            pseudo_outputs1_2 = torch.argmax(outputs_soft2[args.labeled_bs:-1].detach(), dim=1, keepdim=False)
            pseudo_outputsema_2 = torch.argmax(ema_output2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1_2 = ce_loss(outputs2[args.labeled_bs:-1], pseudo_outputsema_2)
            pseudo_supervision2_2 = ce_loss(ema_output2[args.labeled_bs:], pseudo_outputs1_2)
            
            pseudo_outputs1_3 = torch.argmax(outputs_soft3[args.labeled_bs:-1].detach(), dim=1, keepdim=False)
            pseudo_outputsema_3 = torch.argmax(ema_output3[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1_3 = ce_loss(outputs3[args.labeled_bs:-1], pseudo_outputsema_3)
            pseudo_supervision2_3 = ce_loss(ema_output3[args.labeled_bs:], pseudo_outputs1_3)
            
            pseudo_supervision1_total = (pseudo_supervision1_0 + pseudo_supervision1_1+pseudo_supervision1_2+pseudo_supervision1_3)/4
            pseudo_supervision2_total = (pseudo_supervision2_0 + pseudo_supervision2_1+pseudo_supervision2_2+pseudo_supervision2_3)/4

            model1_loss = supervised_loss_1 + consistency_weight * pseudo_supervision1_total
            model2_loss = supervised_loss_ema + consistency_weight * pseudo_supervision2_total
            supervised_loss = model1_loss+model2_loss
#             print(supervised_loss)
#             # focal loss
#             loss_focal = focal_loss(outputs[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())

#             # boundary loss
#             with torch.no_grad():
#                 # defalut using compute_sdf; however, compute_sdf1_1 is also worth to try;
#                 gt_sdf_npy = compute_sdf1_1(label_batch.cpu().numpy(), outputs_soft.shape)
#                 gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs_soft.device.index)
#             loss_bd = boundary_loss(outputs_soft[:args.labeled_bs], gt_sdf[:args.labeled_bs])

            
            # total supervised loss
#             supervised_loss = 0.5 * (loss_ce + loss_dice) + loss_focal + 0.5 * loss_bd


#             Confident Learning - weakly supervised Loss
            
#             noisy labels and images in batch
            noisy_label_batch = label_batch[args.labeled_bs:]
            CL_inputs = unlabeled_volume_batch
            
            if iter_num < 10000:
            # if True:
                loss_ce_weak = 0.0
            elif iter_num >= 10000:
                # Teacher model training after 4000 iteration in order to asure the teacher model had learn st
                with torch.no_grad():
                    out_main,out_main1,out_main2,out_main3 = ema_model(CL_inputs)
                    pred_soft_np = torch.softmax(out_main, dim=1).cpu().detach().numpy() # (bs,2,H,W)
                    pred_soft_np1 = torch.softmax(out_main1, dim=1).cpu().detach().numpy() # (bs,2,H,W)
                    pred_soft_np2 = torch.softmax(out_main2, dim=1).cpu().detach().numpy() # (bs,2,H,W)
                    pred_soft_np3 = torch.softmax(out_main3, dim=1).cpu().detach().numpy() # (bs,2,H,W)

        
                masks_np = noisy_label_batch.cpu().detach().numpy() # (bs,H,W)
                masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)

                # Out main 0
                preds_softmax_np_accumulated = np.swapaxes(pred_soft_np, 1, 2)
                preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
                preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
                preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)
                
                # Out main 1
                preds_softmax_np_accumulated1 = np.swapaxes(pred_soft_np1, 1, 2)
                preds_softmax_np_accumulated1 = np.swapaxes(preds_softmax_np_accumulated1, 2, 3)
                preds_softmax_np_accumulated1 = preds_softmax_np_accumulated1.reshape(-1, num_classes)
                preds_softmax_np_accumulated1 = np.ascontiguousarray(preds_softmax_np_accumulated1)
                
                # Out main 2
                preds_softmax_np_accumulated2 = np.swapaxes(pred_soft_np2, 1, 2)
                preds_softmax_np_accumulated2 = np.swapaxes(preds_softmax_np_accumulated2, 2, 3)
                preds_softmax_np_accumulated2 = preds_softmax_np_accumulated2.reshape(-1, num_classes)
                preds_softmax_np_accumulated2 = np.ascontiguousarray(preds_softmax_np_accumulated2)
                
                # Out main 3
                preds_softmax_np_accumulated3 = np.swapaxes(pred_soft_np3, 1, 2)
                preds_softmax_np_accumulated3 = np.swapaxes(preds_softmax_np_accumulated3, 2, 3)
                preds_softmax_np_accumulated3 = preds_softmax_np_accumulated3.reshape(-1, num_classes)
                preds_softmax_np_accumulated3 = np.ascontiguousarray(preds_softmax_np_accumulated3)

                
                
                # Compare with each elements with order from left to right then go down
                # Eg : (bs,num_classes,W,H) -> (H * W,num_classes) compare with (H*W) of mask
                assert masks_np_accumulated.shape[0] == preds_softmax_np_accumulated.shape[0]
                assert masks_np_accumulated.shape[0] == preds_softmax_np_accumulated1.shape[0]
                assert masks_np_accumulated.shape[0] == preds_softmax_np_accumulated2.shape[0]
                assert masks_np_accumulated.shape[0] == preds_softmax_np_accumulated3.shape[0]

                CL_type = args.CL_type

                try:
                    if CL_type in ['both', 'Qij']:
#                         print(masks_np_accumulated.shape)
#                         print(preds_softmax_np_accumulated.shape)
                        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                                   prune_method='both', n_jobs=1)
                        noise1 = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated1,
                                                                   prune_method='both', n_jobs=1)
                        noise2 = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated2,
                                                                   prune_method='both', n_jobs=1)
                        noise3 = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated3,
                                                                   prune_method='both', n_jobs=1)
                    elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                                   prune_method=CL_type,n_jobs=1)
                        noise1 = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated1,
                                                                   prune_method=CL_type, n_jobs=1)
                        noise2 = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated2,
                                                                   prune_method=CL_type, n_jobs=1)
                        noise3 = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated3,
                                                                   prune_method=CL_type, n_jobs=1)

                    # Guidance to the model which pixel in image maybe in wrong label
                    # True mean at that pixel is labeled wrong and vice versa
                    confident_maps_np = noise.reshape(-1, 320, 320).astype(np.uint8)  # (bs, 320, 320)
                    confident_maps_np1 = noise1.reshape(-1, 320, 320).astype(np.uint8)  # (bs, 320, 320)
                    confident_maps_np2 = noise2.reshape(-1, 320, 320).astype(np.uint8)  # (bs, 320, 320)
                    confident_maps_np3 = noise3.reshape(-1, 320, 320).astype(np.uint8)  # (bs, 320, 320)


                    confident_maps_np_total = 0.4 * confident_maps_np + 0.3 * confident_maps_np1 + 0.2 * confident_maps_np2 + 0.1 * confident_maps_np3
                    confident_maps_np_total = (confident_maps_np_total >= 0.5).astype(np.uint8)
    #                     confident_maps_np_total = confident_maps_np

    #                 print(confident_maps_np_total.shape)
    #                 print(np.unique(confident_maps_np_total))
                    # Correct the label
#                     correct_type = 'smooth'
                    correct_type = 'smooth'
#                     if correct_type == 'smooth':
                    if True:
                        smooth_arg = 0.8
                        corrected_masks_np = masks_np + confident_maps_np_total * np.power(-1, masks_np) * smooth_arg
                        print('Smoothly correct the noisy label')
                    else:
                        corrected_masks_np = masks_np + confident_maps_np_total * np.power(-1, masks_np)

    #                     correct_type = 'uncertainty_smooth'
    #                     if correct_type == 'fixed_smooth':
    #                         smooth_arg = 0.8
    #                         corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
    #                         print('FS correct the noisy label')
    #                     elif correct_type == 'uncertainty_smooth':
    #                         uncertainty_np = uncertainty.cpu().detach().numpy()
    #                         uncertainty_np_squeeze = np.squeeze(uncertainty_np)
    #                         smooth_arg = 1 - uncertainty_np_squeeze
    #                         corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
    #                         print('UDS correct the noisy label')
    #                     else:
    #                         corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
    # 						print('Hard correct the noisy label')

                    noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft.device.index)
                    # print('Shape of noisy_label_batch:', noisy_label_batch.shape)
                    loss_ce_weak = ce_loss(outputs[args.labeled_bs:-1], noisy_label_batch.long())
    #                     print(loss_ce_weak)
                    loss_focal_weak = focal_loss(outputs[args.labeled_bs:-1], noisy_label_batch.long())
                    supervised_loss = supervised_loss + 0.5 * (loss_ce_weak + loss_focal_weak)

                except Exception as e:
                    loss_ce_weak = loss_ce_weak


            # Unsupervised Consistency Loss
            
            if iter_num < 1000:
#             if False:
                consistency_loss_total = 0.0
            else:
                preds_mix = (batch_pred_mixed+batch_pred_mixed1 + batch_pred_mixed2+batch_pred_mixed3)/4
                
                variance_main = torch.sum(kl_distance(
                torch.log(batch_pred_mixed), preds_mix), dim=1, keepdim=True)
                exp_variance_main = torch.exp(-variance_main)

                variance_aux1 = torch.sum(kl_distance(
                    torch.log(batch_pred_mixed1), preds_mix), dim=1, keepdim=True)
                exp_variance_aux1 = torch.exp(-variance_aux1)

                variance_aux2 = torch.sum(kl_distance(
                    torch.log(batch_pred_mixed2), preds_mix), dim=1, keepdim=True)
                exp_variance_aux2 = torch.exp(-variance_aux2)

                variance_aux3 = torch.sum(kl_distance(
                    torch.log(batch_pred_mixed3), preds_mix), dim=1, keepdim=True)
                exp_variance_aux3 = torch.exp(-variance_aux3)


                consistency_dist_main = (
                    preds_mix - batch_pred_mixed) ** 2

                consistency_loss_main = torch.mean(
                    consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)

                consistency_dist_aux1 = (
                    preds_mix - batch_pred_mixed1) ** 2
                consistency_loss_aux1 = torch.mean(
                    consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

                consistency_dist_aux2 = (
                    preds_mix - batch_pred_mixed2) ** 2
                consistency_loss_aux2 = torch.mean(
                    consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

                consistency_dist_aux3 = (
                    preds_mix - batch_pred_mixed3) ** 2
                consistency_loss_aux3 = torch.mean(
                    consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

                consistency_loss_ema = (consistency_loss_main + consistency_loss_aux1 +
                                    consistency_loss_aux2 + consistency_loss_aux3) / 4
                consistency_loss_mix = torch.mean((outputs_soft[-1] - batch_pred_mixed) ** 2)
                consistency_loss1_mix = torch.mean((outputs_soft1[-1] - batch_pred_mixed1) ** 2)
                consistency_loss2_mix = torch.mean((outputs_soft2[-1] - batch_pred_mixed2) ** 2)
                consistency_loss3_mix = torch.mean((outputs_soft3[-1] - batch_pred_mixed3) ** 2)
                
#                 consistency_loss = torch.mean((outputs_soft[args.labeled_bs:-1] - ema_output_soft) ** 2)
#                 consistency_loss1 = torch.mean((outputs_soft1[args.labeled_bs:-1] - ema_output_soft1) ** 2)
#                 consistency_loss2 = torch.mean((outputs_soft2[args.labeled_bs:-1] - ema_output_soft2) ** 2)
#                 consistency_loss3 = torch.mean((outputs_soft3[args.labeled_bs:-1] - ema_output_soft3) ** 2)
#                 print(consistency_loss)
#                 print(consistency_loss1)
#                 print(consistency_loss2)
#                 print(consistency_loss3)
#                 print(consistency_loss.item())
#                 print(consistency_loss1.item())
#                 print(consistency_loss.item() + consistency_loss1.item() + consistency_loss2.item() + consistency_loss3.item())
#                 consistency_loss_total = consistency_loss.item() + consistency_loss1.item() + consistency_loss2.item() + consistency_loss3.item()
                consistency_loss_2model = (consistency_loss_mix+consistency_loss1_mix+consistency_loss2_mix+consistency_loss3_mix)/4
                consistency_loss_total = consistency_loss_ema+consistency_loss_2model

#                 print(consistency_loss_total)
            # Total Loss = Supervised + Consistency
            loss = supervised_loss + consistency_weight *consistency_loss_total

            optimizer.zero_grad()
            optimizer_ema.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_ema.step()
#             update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_ema.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

#             logging.info(
#                 'iteration %d : loss : %f, loss_ce: %f, loss_ce1: %f, loss_ce2 : %f, loss_ce3 : %f, loss_dice: %f, loss_dice1: %f, loss_consistency: %f, loss_weak: %f' %
#                 (iter_num, loss.item(), loss_ce.item(), loss_ce1.item(), loss_ce2.item(), loss_ce3.item(), loss_dice.item(), loss_dice1.item(), consistency_loss_total, loss_ce_weak))

            logging.info(
                'iteration %d : loss: %f, loss model 1: %f, loss model 2: %f, loss_consistency: %f, loss_weak: %f' %
                (iter_num, loss.item(), model1_loss.item(), model2_loss.item(), consistency_loss_total, loss_ce_weak))
    
#             logging.info(
#                 'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_consistency: %f' %
#                 (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_loss_total))
#             print('-'*50)

            # Validation
            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_concat_volume(
                        sampled_batch["image"], sampled_batch["label_ROI"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model_val_mean_dice', performance, iter_num)
                writer.add_scalar('info/model_val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'fold_{}'.format(fold_num),
                                                  'model_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, 'fold_{}'.format(fold_num),
                                             '{}_best_model_dice_{}.pth'.format(args.model,round(best_performance, 4)))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : model_mean_dice : %f model_mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
                
                ema_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_concat_volume(
                        sampled_batch["image"], sampled_batch["label_ROI"], ema_model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/ema_model_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/ema_model_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance_ema = np.mean(metric_list, axis=0)[0]

                mean_hd95_ema = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/ema_model_val_mean_dice', performance_ema, iter_num)
                writer.add_scalar('info/ema_model_val_mean_hd95', mean_hd95_ema, iter_num)

                if performance_ema > best_performance_ema:
                    best_performance_ema = performance_ema
                    save_mode_path = os.path.join(snapshot_path, 'fold_{}'.format(fold_num),
                                                  'ema_model_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance_ema, 4)))
                    save_best = os.path.join(snapshot_path, 'fold_{}'.format(fold_num),
                                             '{}_best_ema_model_dice_{}.pth'.format(args.model,round(best_performance_ema, 4)))
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance_ema, mean_hd95_ema))
                ema_model.train()


#             if iter_num % 3000 == 0:
#                 save_mode_path = os.path.join(
#                     snapshot_path,'fold_{}'.format(fold_num), 'iter_' + str(iter_num) + '.pth')
#                 torch.save(model.state_dict(), save_mode_path)
#                 logging.info("save model to {}".format(save_mode_path))
            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_ema.param_groups:
                    param_group['lr'] = lr_
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


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

    add_ex = ''
    snapshot_path = "../model/{}_labeled/{}{}_kfold".format(
        args.exp, args.model,add_ex)
    print(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

#     for fold_num in range(1,6):
#         if fold_num != 5:
#             continue
#         logging.basicConfig(filename=snapshot_path + "/log_fold_{}.txt".format(fold_num), level=logging.INFO,
#                             format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#         logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#         logging.info(str(args))
#         train(args, snapshot_path,fold_num)

    if args.fold == None:
        for fold_num in range(1,6):
            print('The automatical fold number is: ',fold_num)
            logging.basicConfig(filename=snapshot_path + "/log_fold_{}.txt".format(fold_num), level=logging.INFO,
                                format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            logging.info(str(args))
            train(args, snapshot_path,args.fold)
    else:
        print('The manual fold number is:',args.fold)
        logging.basicConfig(filename=snapshot_path + "/log_fold_{}.txt".format(args.fold), level=logging.INFO,
                                format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        train(args, snapshot_path,args.fold)
