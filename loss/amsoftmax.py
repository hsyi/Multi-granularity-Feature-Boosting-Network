import torch
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmax(nn.Module):
    def __init__(self, args):
        super(AMSoftmax, self).__init__()
        self.feats = args.feats
        self.class_num = args.num_classes
        print(self.class_num)
        self.weight = Parameter(torch.Tensor(self.class_num, self.feats))
        self.weight.data.uniform_(-1, 1).renorm(2, 1, 1e-5).mul_(1e5)
        print(self.weight.data.cpu())

    def forward(self, x, target, scale=10, margin=0.):
        # self.it += 1
        w = self.weight
        w_norm = F.normalize(w, dim=1)
        x_norm = x.norm(p=2, dim=1)
        cos_theta = x.mm(w_norm.t())
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= margin
        output = output * scale

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

