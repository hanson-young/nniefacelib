import torch
from torch import nn
import torch.nn.functional as F
import math

class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        mat_ratio = torch.Tensor([
            1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        ]).cuda()
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)
        pose_loss = torch.sum((euler_angle_gt - angle) * (euler_angle_gt - angle), dim=1)
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        return torch.mean(weight_angle * weight_attribute * l2_distant) * 100, torch.mean(pose_loss) * 100

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), dim=1)
        pose_loss = torch.sum((euler_angle_gt - angle) * (euler_angle_gt - angle), dim=1)
        return torch.mean(l2_distant) * 100, \
               torch.mean(pose_loss) * 1000

class SmoothL1(nn.Module):
    def __init__(self):
        super(SmoothL1, self).__init__()
        self.num_lds = 98
        self.size = self.num_lds * 2

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):
        landm_p = landmarks.view(-1, self.size)
        landm_t = landmark_gt.view(-1, self.size)
        lds_loss = F.smooth_l1_loss(landm_p, landm_t, reduction='mean')
        pose_p = angle.view(-1, 3)
        pose_t = euler_angle_gt.view(-1, 3)
        pose_loss = F.smooth_l1_loss(pose_p, pose_t, reduction='mean')

        return lds_loss * 1000, pose_loss * 1000


class WingLoss(nn.Module):
    def __init__(self):
        super(WingLoss, self).__init__()
        self.num_lds = 98
        self.size = self.num_lds * 2
        self.w = 10.0
        self.epsilon = 2.0

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):
        landm_p = landmarks.reshape(-1, self.num_lds, 2)
        landm_t = landmark_gt.reshape(-1, self.num_lds, 2)
        x = landm_t - landm_p
        c = self.w * (1.0 - math.log(1.0 + self.w / self.epsilon))
        absolute_x = torch.abs(x)
        lds_losses = torch.where(self.w > absolute_x, self.w * torch.log(1.0 + absolute_x / self.epsilon), absolute_x - c)
        lds_loss = torch.mean(torch.sum(lds_losses, axis=[1, 2]), axis=0)

        pose_p = angle.view(-1, 3)
        pose_t = euler_angle_gt.view(-1, 3)
        pose_loss = F.smooth_l1_loss(pose_p, pose_t, reduction='mean')

        return lds_loss, pose_loss*1000