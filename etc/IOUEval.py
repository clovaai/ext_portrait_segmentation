import torch
import numpy as np

#adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class iouEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.total_hist = np.zeros([self.nClasses,self.nClasses])

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses) & (b < self.nClasses)
        # print(np.unique(a[k]))
        # print(np.unique(b[k]))
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()

        hist = self.compute_hist(predict, gth)
        self.total_hist += hist

    def getMetric(self):
        if self.nClasses == 20:
            hist = self.total_hist[0:19,0:19]
        else:
            hist = self.total_hist
        epsilon = 0.00000001

        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iu)

        return overall_acc, per_class_acc, per_class_iu, mIou