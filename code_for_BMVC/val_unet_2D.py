import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from skimage import measure
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
def calculate_metric_percase(pred, gt):
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


# ---------------------------IRCAD----------------------
def test_single_concat_volume(image, label, net, classes):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() #unsqueeze to remove the batch size axis
    prediction = np.zeros_like(label)
    for ind in range(int(image.shape[0]/2)):
        slice = image[ind, :, :] 
        prob_slice = image[int(image.shape[0]/2)+ind, :, :]
        img = np.expand_dims(slice, axis=0)
        prob_ = np.expand_dims(prob_slice, axis=0)
        concat_input = np.concatenate((img, prob_), axis=0) # (2, H, W)
        input = torch.from_numpy(concat_input).unsqueeze(0).float().cuda()
        net.eval()
        # start predict
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    if True:
        print('preprocessing')
        prediction = post_processing(prediction)
    metric_list = []
    for i in range(1, classes):
        prediction[prediction > 0] = 1
        label[label > 0] = 1
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
