import torch.nn as nn
import torch.nn.functional as F
import torch

class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, weight=None, ignore = None):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        '''

        super().__init__()
        if int(torch.__version__[2]) < 4:
            self.loss = nn.NLLLoss2d(weight, ignore_index=ignore)
        else:
            self.loss = nn.NLLLoss(weight, ignore_index=ignore)

    def forward(self, outputs, targets):
        # print(torch.unique(targets))
        return self.loss(F.log_softmax(outputs, 1), targets)