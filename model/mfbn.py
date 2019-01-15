import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

def make_model(args):
    return MFBN(args.num_classes, args.feats, args.pool)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                                nn.Linear(channel, channel // reduction ),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel ),
                                nn.Sigmoid()
                    )
        #self._initialize_weights(self.fc)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MFBN(nn.Module):
    def __init__(self, num_classes, feats, pool='avg'):
        super(MFBN, self).__init__()

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))), #no downsample
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        reductions = 8
        self.att1 = SELayer(1024, reductions)
        self.att2 = SELayer(1024, reductions)
        self.att3 = SELayer(1024, reductions)

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        if pool == 'max':
            pool2d = nn.MaxPool2d
        elif pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        #self.reduction = nn.Sequential(nn.Conv2d(2048*8, 2048, 1, bias=False), nn.BatchNorm2d(2048), nn.LeakyReLU(0.1))
        reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.LeakyReLU(0.1))
        # reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU(),nn.Dropout(p=0.5))

        #self._init_reduction(self.reduction)
        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        self.fc_id_2048 = nn.Linear(2048, num_classes)

        self._init_fc(self.fc_id_2048)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        x1 = self.att1(x)
        x2 = self.att2(x) + x1
        x3 = self.att3(x) + x2

        p1 = self.p1(x1)
        p2 = self.p2(x2)
        p3 = self.p3(x3)

        p2 = p2 + p1
        p3 = p3 + p2 + p1


        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        #f_all = self.reduction(torch.cat([zg_p1, zg_p2, zg_p3, z0_p2, z1_p2, z0_p3, z1_p3, z2_p3], dim=1)).squeeze(dim=3).squeeze(dim=2)
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        predict = f_all
        l_predict = self.fc_id_2048(f_all)
        '''
        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        l_predict = self.fc_id_2048(predict)

        #return predict
        return predict, l_predict


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import argparse
    parser = argparse.ArgumentParser()
    from torchsummary import summary
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    parser.add_argument('--pool', type=str, default='avg', help='pool function')
    parser.add_argument('--feats', type=int, default=256, help='number of feature maps')
    #parser.add_argument('--height', type=int, default=384, help='height of the input image')
    #parser.add_argument('--width', type=int, default=128, help='width of the input image')
    parser.add_argument('--num_classes', type=int, default=751, help='')
    args = parser.parse_args()
    net = MGN(args)
    net.to('cuda')
    import time
    net.eval()
    start_time = time.time()
    for i in range(100):
        x = torch.randn(20, 3, 384, 128)
        x = x.to('cuda')
        y = net(x)
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost time:{}, average:{}".format(cost_time, cost_time/100))

    #summary(model, (3, 384, 128))


