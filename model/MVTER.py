import torch
import torch.nn as nn
from torchvision.models import googlenet
from tqdm import tqdm

import model.InceptionV4 as Icpv4

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias) 

class MVTER(nn.Module):
    def __init__(self, nclasses=33, m=12, cnn_name='googlenet'):
        super(MVTER, self).__init__()
        # number of view points
        self.m = m
        # GoogLeNet backbone
        assert(cnn_name == 'googlenet')
        # self.E = nn.Sequential(*list(googlenet(pretrained=True).children())[:-1])
        self.E = GVCNN()
        # transformation decoding
        self.D = nn.Linear(3072, 3)
        # classificaton
        self.T = nn.Linear(1536, nclasses)
        
    def fusion(self, multiple_view_feature):
        # list of (batch_size * feature_dim)
        f = [torch.unsqueeze(fi, dim = 0) for fi in multiple_view_feature]
        f = torch.cat(f, dim = 0)
        pooled, _ = torch.max(f, dim = 0)
        return pooled
    
    def forward(self, im1, im2):
        # f1 = self.fusion([torch.flatten(self.E(im1[:, i, ...]), start_dim=1) for i in range(self.m)])
        # f2 = self.fusion([torch.flatten(self.E(im1[:, i, ...]), start_dim=1) for i in range(self.m)])
        f1 = self.E(im1)
        f2 = self.E(im2)
        f = torch.cat((f1, f2), dim = 1)
        return self.T(f1), self.D(f)

class GroupSchema(nn.Module):
    """
    differences from paper:
    1. Considering the amount of params, we use 1*1 conv instead of  fc
    2. if the scores are all very small, it will cause a big problem in params' update,
    so we add a softmax layer to normalize the scores after the convolution layer
    """
    def __init__(self):
        super(GroupSchema, self).__init__()
        self.score_layer = OneConvFc()
        self.sft = nn.Softmax(dim=1)

    def forward(self, raw_view):
        """
        :param raw_view: [V N C H W]
        :return:
        """
        scores = []
        for batch_view in raw_view:
            # batch_view: [N C H W]
            # y: [N]
            y = self.score_layer(batch_view)
            y = torch.sigmoid(torch.log(torch.abs(y)))
            scores.append(y)
        # view_scores: [N V]
        view_scores = torch.stack(scores, dim=0).transpose(0, 1)
        view_scores = view_scores.squeeze(dim=-1)
        return self.sft(view_scores)

def group_pool(final_view, scores):
    """
    view pooling + group fusion
    :param final_view: # [N V C H W]
    :param scores: [N V] scores
    :return: shape descriptor
    """
    shape_descriptors = []

    for idx, (ungrp_views, view_scores) in enumerate(zip(final_view, scores)):
        # ungrp_views: [V C H W]
        # view_scores: [V]

        # view pooling
        shape_descriptors.append(view_pool(ungrp_views, view_scores))
    # [N C H W]
    y = torch.stack(shape_descriptors, 0)
    # print('[2 C H W]', y.size())
    return y

class OneConvFc(nn.Module):
    """
    1*1 conv + fc to obtain the grouping schema
    """
    def __init__(self):
        super(OneConvFc, self).__init__()
        self.conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, stride=1)
        self.fc = nn.Linear(in_features=52*52, out_features=1)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def view_pool(ungrp_views, view_scores, num_grps=7):
    """
    :param ungrp_views: [V C H W]
    :param view_scores: [V]
    :param num_grps the num of groups. used to calc the interval of each group.
    :return: grp descriptors [(grp_descriptor, weight)]
    """

    def calc_scores(scores):
        """
        :param scores: [score1, score2 ....]
        :return:
        """
        n = len(scores)
        s = torch.ceil(scores[0]*n)
        for idx, score in enumerate(scores):
            if idx == 0:
                continue
            s += torch.ceil(score*n)
        s /= n
        return s

    interval = 1 / (num_grps + 1)
    # begin = 0
    view_grps = [[] for i in range(num_grps)]
    score_grps = [[] for i in range(num_grps)]

    for idx, (view, view_score) in enumerate(zip(ungrp_views, view_scores)):
        begin = 0
        for j in range(num_grps):
            right = begin + interval
            if j == num_grps-1:
                right = 1.1
            if begin <= view_score < right:
                view_grps[j].append(view)
                score_grps[j].append(view_score)
            begin += interval
    # print(score_grps)
    view_grps = [sum(views)/len(views) for views in view_grps if len(views) > 0]
    score_grps = [calc_scores(scores) for scores in score_grps if len(scores) > 0]

    shape_des = map(lambda a, b: a*b, view_grps, score_grps)
    shape_des = sum(shape_des)/sum(score_grps)

    # !!! if all scores are very small, it will cause some problems in params' update
    if sum(score_grps) < 0.1:
        # shape_des = sum(view_grps)/len(score_grps)
        print(sum(score_grps), score_grps)
    # print('score total', score_grps)
    return shape_des

class GVCNN(nn.Module):
    def __init__(self, nclasses=33, num_views=12):
        super(GVCNN, self).__init__()

        self.nclasses = nclasses
        self.num_views = num_views

        model = Icpv4.inceptionv4()

        # first six layers of inception_v4
        self.fcn_1 = nn.Sequential(*list(model.features[0:5]))

        # grouping module
        self.group_schema = GroupSchema()
        init_weights(self.group_schema)

        # remain layers of inception_v4
        self.fcn_2 = nn.Sequential(*list(model.features[5:]))

        self.avg_pool_2 = nn.AvgPool2d(kernel_size=5, stride=5, padding=0)
        self.fc_2 = model.last_linear

    def forward(self, x):
        """
        :param x: N V C H W
        :return:
        """
        # transform the x from [N V C H W] to [NV C H W]
        x = x.view((int(x.shape[0] * self.num_views), x.shape[-3], x.shape[-2], x.shape[-1]))
        # print('[24 3 224 224]', x.size())

        # [NV 192 52 52]
        y = self.fcn_1(x)
        # print('[24 192 52 52]', y.size())

        # [N V 192 52 52]
        y1 = y.view(
            (int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))  # (8,12,512,7,7)
        # print('[2 12 192 52 52]', y1.size())

        # [V N 192 52 52]
        raw_view = y1.transpose(0, 1)
        # print('[12, 2, 192, 52, 52]', raw_view.size())

        # [N V] scores
        view_scores = self.group_schema(raw_view)
        # print('[2 12]', view_scores.size())

        # [NV 1536 5 5]
        final_view = self.fcn_2(y)
        # print('[24 1536 5 5]', final_view.size())

        # [N V C H W]
        final_view = final_view.view(
            (int(final_view.shape[0]/self.num_views)),
            self.num_views, final_view.shape[-3],
            final_view.shape[-2], final_view.shape[-1]
        )
        # print('[2 12 1536 5 5]', final_view.size())

        # [N C H W]
        shape_decriptors = group_pool(final_view, view_scores)

        # print(shape_decriptors.size())
        # print(self.avg_pool_2)

        z = self.avg_pool_2(shape_decriptors)
        z = z.view(z.size(0), -1)
        # print(z.shape)
        return z


if __name__ == '__main__':
    num_of_classes = 33
    X = torch.randn(24, 3, 224, 224)
    X1 = torch.randn(2, 12, 3, 224, 224)
    X2 = torch.randn(2, 12, 3, 224, 224)
    # label = (torch.rand(24) * 33).long()
    # euler = torch.randn(24, 3)

    # mvter = MVTER()
    # loss_m = nn.MSELoss()
    # loss_task = nn.CrossEntropyLoss()
    # w = 1.0
    # pred_labels, pred_eulers = mvter(X1, X2)
    # print(loss_task(pred_labels, label) + w * loss_m(pred_eulers, euler))

    mvter = MVTER()
    print(mvter(X1, X2))
