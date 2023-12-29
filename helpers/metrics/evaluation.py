import torch
import numpy as np
import torchvision
from skimage import morphology
# SR : Segmentation Result
# GT : Ground Truth


def get_accuracy(SR, GT, threshold=0.5):#报错loss未定义，是否和梯度有关，存在影响？？？？
    SR = SR > threshold#在所有预测样本中，预测准确的概率
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2)
    acc = float(corr) / float(tensor_size)
    return acc

def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall 所有正例样本中被正确预测的概率，衡量对正类样本的识别能力
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR=SR+0
    GT=GT+0
    TP = SR * GT
    FN = (1-SR) * GT
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    return SE

def get_skeleton_recall(SR, GT, device, threshold=0.5):
    # Skeleton Recall 骨架 recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR=SR+0
    GT=GT+0
    # #可视化骨架
    # img1 = GT.cpu()
    # img1 = img1.squeeze(0)
    # img1 = img1.numpy()
    # img1 = morphology.skeletonize(img1)
    # from PIL import Image
    # img1 = Image.fromarray(img1)
    # img1.save("out.png")
    img = GT.cpu().numpy()
    img = morphology.skeletonize(img)
    GT = torch.tensor(img)
    GT = GT.to(device)
    Inter = torch.sum((SR + GT) == 2)
    SRe = float(Inter) / (float(torch.sum(GT)) + 1e-6)
    return SRe

def get_skeleton_precision(SR, GT, device, threshold=0.5):
    # Skeleton precision 骨架 recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR=SR+0
    GT=GT+0
    img = SR.cpu().numpy()
    img = morphology.skeletonize(img)
    SR = torch.tensor(img)
    SR = SR.to(device)
    Inter = torch.sum((SR + GT) == 2)
    SPr = float(Inter) / (float(torch.sum(SR)) + 1e-6)
    return SPr

def get_specificity(SR, GT, threshold=0.5):#所有负例样本中被正确预测的概率，衡量对负类样本的识别能力
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR + 0
    GT = GT + 0
    TN = (1-SR) * (1-GT)
    FP = SR * (1-GT)
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP

def get_precision(SR, GT, threshold=0.5):#在全部阳性预测中，正确预测结果占的比例
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR + 0
    GT = GT + 0
    TP = SR * GT
    FP = SR * (1-GT)
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
    return PC

def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)
    F1 = 2 * SE * PC / (SE + PC + 1e-6)
    return F1

def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity #集合之间的相似度，值越大，样本的相似度越高
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR + 0
    GT = GT + 0
    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)
    JS = float(Inter) / (float(Union) + 1e-6)
    return JS

def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR + 0
    GT = GT + 0
    Inter = torch.sum((SR + GT) == 2)
    # Inter = torch.sum(SR * GT)
    outer=torch.sum(SR) + torch.sum(GT)
    DC = np.float(2 * Inter) / (np.float(outer) + 1e-6)
    return DC