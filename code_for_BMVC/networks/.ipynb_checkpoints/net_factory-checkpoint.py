from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS
# MONAI
from networks.UNet_3Plus import UNet_3Plus
from networks.UNet_2Plus import UNet_2Plus
from monai.networks.nets import SegResNet, VNet, UNETR
from networks.model import DcUnet
from networks.ae import kiunet,reskiunet,densekiunet
from networks.MTUNet import MTUNet
from networks.model_shadow import shadow_aug, ShadowUNet
from networks.unet_urpc import UNet_URPC, UNet_CCT
import torch

print('Gpu available:',torch.cuda.is_available())
def net_factory(net_type="unet", in_chns=2, class_num=2):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet3+":
        net = UNet_3Plus().cuda()
    elif net_type == "DCUnet":
        net = DcUnet(input_channels=2).cuda()
    elif net_type == "unet2+":
        net = UNet_2Plus().cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "kiunet":
        net = kiunet().cuda()
    elif net_type == "reskiunet":
        net = reskiunet().cuda()
    elif net_type == "densekiunet":
        net = densekiunet().cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "segresnet":
        net = SegResNet(
                spatial_dims=2, 
                init_filters = 16,
                in_channels=in_chns, 
                out_channels=class_num,
                dropout_prob = 0.2,
                norm_name = 'group'
                ).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()   
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "UnetR":
        net = UNETR(in_channels=in_chns, out_channels=class_num, img_size=320, feature_size=96, spatial_dims=2, pos_embed='conv', norm_name='instance').cuda()
    elif net_type == "MTUnet":
        net = MTUNet().cuda()
    elif net_type == "shadow":
        net = ShadowUNet(in_ch=in_chns, base_ch=64).cuda()
    elif net_type == "vnet":
        net = VNet().cuda()
    else:
        net = None
    return net
