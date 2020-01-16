import torch
from etc.IOUEval import iouEval
import time
import os
import torchvision
import numpy as np
from etc.lovasz_losses import iou_binary, calcF1
import cv2
import datetime

def val_edge(num_gpu, val_loader, model, criterion, lovasz, vis=False):
    '''
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to evaluation mode
    model.eval()

    total_time = 0
    epoch_loss = []
    epoch_lossE = []
    total_batches = len(val_loader)
    rand_pick = np.random.randint(0, len(val_loader))
    mIOU = []
    f1score=[]
    for i, (input, target, edge_target) in enumerate(val_loader):
        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()
            edge_target = edge_target.cuda()

            with torch.no_grad():
                input_var, target_var, edge_target_var \
                    = torch.autograd.Variable(input), torch.autograd.Variable(target), \
                      torch.autograd.Variable(edge_target)

                # run the mdoel
                output = model(input_var)

                # compute the loss
                loss = criterion(output, target_var)
                lossE = 0.5*criterion(output, edge_target_var)

        epoch_loss.append(loss.item())
        epoch_lossE.append(lossE.item())

        if lovasz:
            IOU = iou_binary((output.data > 0).long(), target_var)
            f1 =  calcF1((output.data > 0).long(), edge_target_var, ignore=255, per_image=True)
        else:
            IOU = iou_binary(output.max(1)[1].data.long(), target_var)
            f1 =  calcF1(output.max(1)[1].data.long(), edge_target_var, ignore=255, per_image=True)


        mIOU.append(IOU)
        f1score.append(f1)

        time_taken = time.time() - start_time
        total_time += time_taken
        # compute the confusion matrix

        if i == rand_pick and vis:
            save_input = input_var
            save_est = output
            save_target = edge_target_var

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    average_epoch_lossE_val = sum(epoch_lossE) / len(epoch_lossE)

    print('loss: %.3f  lossE: %.3f  time:%.2f  f1score: %.2f'   % (
    average_epoch_loss_val, average_epoch_lossE_val, total_time / total_batches, sum(f1score) / len(f1score) ))

    if vis:
        return average_epoch_loss_val, average_epoch_lossE_val, sum(mIOU) / len(mIOU), save_input, save_est, save_target
    else:
        return average_epoch_loss_val, average_epoch_lossE_val, sum(mIOU) / len(mIOU)

def train_edge(num_gpu, train_loader, model, criterion, optimizer, lovasz, epoch, total_ep):
    '''

    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    mIOU = []
    epoch_loss = []
    epoch_lossE = []

    total_time = 0
    total_batches = len(train_loader)
    for i, (input, target, edge_target) in enumerate(train_loader):
        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()
            edge_target = edge_target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        edge_target_var = torch.autograd.Variable(edge_target)

        # run the mdoel
        output = model(input_var)

        # set the grad to zero
        optimizer.zero_grad()


        loss = criterion(output, target_var)
        lossE = 0.5*criterion(output, edge_target_var)

        total_loss = loss + lossE

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_lossE.append(lossE.item())

        if lovasz:
            IOU = iou_binary((output.data > 0).long(), target_var)
        else:
            IOU = iou_binary(output.max(1)[1].data.long(), target_var)
        mIOU.append(IOU)

        time_taken = time.time() - start_time
        total_time += time_taken
        # compute the confusion matrix

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    average_epoch_lossE_train = sum(epoch_lossE) / len(epoch_lossE)

    print('[%d/%d] loss: %.3f   lossE: %.3f  time:%.2f' % (
    epoch + 1, total_ep, average_epoch_loss_train, average_epoch_lossE_train, total_time / total_batches))

    return average_epoch_loss_train, average_epoch_lossE_train, sum(mIOU) / len(mIOU),


def train(num_gpu, train_loader, model, criterion, optimizer, lovasz, epoch, total_ep):
    '''

    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    mIOU =[]
    epoch_loss = []
    total_time= 0
    total_batches = len(train_loader)
    for i, (input, target) in enumerate(train_loader):

        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        # set the grad to zero
        optimizer.zero_grad()


        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        time_taken = time.time() - start_time
        total_time += time_taken
        # compute the confusion matrix
        if lovasz:
            IOU = iou_binary((output.data > 0).long(), target_var)
        else:
            IOU = iou_binary(output.max(1)[1].data.long(), target_var)

        mIOU.append(IOU)

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    print('[%d/%d] loss: %.3f time:%.2f' % (epoch+1, total_ep, average_epoch_loss_train, total_time/total_batches))

    return average_epoch_loss_train, sum(mIOU)/len(mIOU),


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)


def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p
        # print(total_paramters)

    return total_paramters


def info_setting(save_dir, model_name, Nparam, Flop):
    now = datetime.datetime.now()
    time_str = now.strftime("%m-%d_%H%M")
    this_savedir = os.path.join(save_dir, model_name+time_str)
    if not os.path.isdir(this_savedir):
        os.mkdir(this_savedir)

    logFileLoc = this_savedir + "/trainValLog.txt"

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(Nparam)))
        logger.write("FLOP: %s" % (str(Flop)))

        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (
            'Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val)', 'lr', 'time'))
    return logger, this_savedir



def colormap_cityscapes(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([128, 64, 128])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])

    return cmap

class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
