import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from utils import losses
from utils.config import args
from Models.STN import SpatialTransformer
from Models.TransMatch import TransMatch
import scipy.spatial

def compute_label_dice(gt, pred):
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

def compute_label_hd(gt, pred):
    gt = np.asarray(gt > 0, dtype=np.bool_)
    pred = np.asarray(pred > 0, dtype=np.bool_)

    if not np.any(gt) or not np.any(pred):
        return np.nan  # 如果有一个为空，返回nan

    gt_points = np.stack(np.nonzero(gt), axis=1)
    pred_points = np.stack(np.nonzero(pred), axis=1)

    # 用KDTree分别查找最近距离
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)

    # gt到pred的最近距离
    gt2pred_dist, _ = tree_pred.query(gt_points, k=1)
    # pred到gt的最近距离
    pred2gt_dist, _ = tree_gt.query(pred_points, k=1)

    hd95_1 = np.percentile(gt2pred_dist, 95)
    hd95_2 = np.percentile(pred2gt_dist, 95)
    hd95 = max(hd95_1, hd95_2)
    return hd95

def test_checkpoint(checkpoint_path):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # 读入fixed图像
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    input_fixed_eval = torch.from_numpy(input_fixed).to(device).float()
    fixed_label = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz"))
    )[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()

    # 加载模型
    net = TransMatch(args).to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    STN.eval()
    STN_label.eval()

    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))
    DSC = []
    HD = []
    for file in test_file_lst:
        name = os.path.split(file)[1]
        input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float()
        label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
        input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))

        pred_flow = net(input_fixed_eval, input_moving)
        pred_label = STN_label(fixed_label, pred_flow)

        dice = compute_label_dice(input_label, pred_label[0, 0, ...].cpu().detach().numpy())
        hd = compute_label_hd(input_label, pred_label[0, 0, ...].cpu().detach().numpy())
        print("File: {}, DSC: {:.4f}, HD: {:.4f}".format(name, dice, hd))
        DSC.append(dice)
        HD.append(hd)
        del pred_flow, pred_label, input_moving

    print("Mean DSC: {:.4f}, Std DSC: {:.4f}".format(np.mean(DSC), np.std(DSC)))
    print("Mean HD95: {:.4f}, Std HD95: {:.4f}".format(np.nanmean(HD), np.nanstd(HD)))

if __name__ == "__main__":
    checkpoint_path = '/mnt/data3/ykb/TransMatch_TMI/Log/dsc0.7020epoch356.pth.tar'  # 修改为你的checkpoint路径
    test_checkpoint(checkpoint_path)