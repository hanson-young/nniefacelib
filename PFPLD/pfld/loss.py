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
        self.s = 5.0

        self.eye_index = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
        # self.pts_68_to_98 = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,33,34,35,36,37,42,43,44,45,46,51,52,53,54,55,56,57,58,59,60,61,63,64,65,67,68,69,71,72,73,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]
        self.pts_68_to_98 = [33,34,35,36,37,42,43,44,45,46,51,52,53,54,55,56,57,58,59,60,64,68,72,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]
        self.pts_onehot = [i for i in range(98)]
        for i in self.pts_onehot:
            if i in self.pts_68_to_98:
                self.pts_onehot[i] = True
            else:
                self.pts_onehot[i] = False
        self.epsilon = 2.0
    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, type_flag , angle, landmarks, train_batchsize):
        landms_const = torch.tensor(-2).cuda()
        pose_68landms_const = torch.tensor(0).cuda()
        # only 98 pts
        pos1 = type_flag == landms_const
        landm_p = landmarks.reshape(-1, self.num_lds, 2)[pos1]
        landm_t = landmark_gt.reshape(-1, self.num_lds, 2)[pos1]
        lds_98_loss = 0
        if landm_p.shape[0] > 0:
            x = landm_t - landm_p
            c = self.w * (1.0 - math.log(1.0 + self.w / self.epsilon))
            absolute_x = torch.abs(x)
            weight_attribute = landm_p*0.0 + 1.0
            weight_attribute[:,self.eye_index] *= self.s
            absolute_x = torch.mul(absolute_x, weight_attribute)
            lds_losses = torch.where(self.w > absolute_x, self.w * torch.log(1.0 + absolute_x / self.epsilon), absolute_x - c)
            lds_98_loss = torch.mean(torch.sum(lds_losses, axis=[1, 2]), axis=0)

        pos2 = type_flag == pose_68landms_const
        pose_p = angle.view(-1, 3)[pos2]
        pose_t = euler_angle_gt.view(-1, 3)[pos2]
        pose_loss = 0
        if pose_p.shape[0] > 0:
            pose_loss = F.smooth_l1_loss(pose_p, pose_t, reduction='mean')

        landm_p = landmarks.reshape(-1, self.num_lds, 2)[pos2]
        landm_t = landmark_gt.reshape(-1, self.num_lds, 2)[pos2]
        lds_68_loss = 0
        if landm_p.shape[0] > 0:
            landm_p = landm_p[:, self.pts_onehot]
            landm_t = landm_t[:, self.pts_onehot]
            x = landm_t - landm_p
            absolute_x = torch.abs(x)
            c = self.w * (1.0 - math.log(1.0 + self.w / self.epsilon))
            lds_losses = torch.where(self.w > absolute_x, self.w * torch.log(1.0 + absolute_x / self.epsilon), absolute_x - c)
            lds_68_loss = torch.mean(torch.sum(lds_losses, axis=[1, 2]), axis=0)

        return lds_98_loss + lds_68_loss, pose_loss*1000