import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import glob

import json

from PIL import Image as PILImage
import importlib
from torchvision.transforms import functional as F

import pickle

from argparse import ArgumentParser
from etc.utils import *
# torch.backends.cudnn.benchmark=True

pallete = [[128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],
           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],
           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],
           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def evaluateModelCV(model, savedir, saveloc, mean, std, imgW, imgH, videoName, Lovasz):
    # gloabl mean and std values
    syn_bg = cv2.imread(os.path.join(savedir, 'syn_bg.jpg'))
    videoOut = os.path.join(saveloc, videoName)
    print("video is saved in " + videoOut)

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    success = True
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
        success = False

    while (success):
        success, img = cap.read()
        if not success or img is None:
            vidcap.release()
            break

        img_orig = np.copy(img)
        # PILImage.fromarray(img_orig).show()

        img = cv2.resize(img, (imgW, imgH))
        # PILImage.fromarray(img).show()

        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]

        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension


        with torch.no_grad():
            img_variable = torch.autograd.Variable(img_tensor)

            if torch.cuda.is_available():
                img_variable = img_variable.cuda()


            img_out = model(img_variable)
        img_orig = cv2.resize(img_orig, (imgW, imgH))

        if Lovasz:
            classMap_numpy = (img_out[0].data.cpu() > 0).numpy()[0]
        else:
            classMap_numpy = img_out[0].max(0)[1].byte().data.cpu().numpy()

        idx_fg = (classMap_numpy == 1)

        syn_bg = cv2.resize(syn_bg, (img_out.size(3), img_out.size(2)))
        img_orig = cv2.resize(img_orig, (img_out.size(3), img_out.size(2)))

        seg_img = 0 * img_orig
        seg_img[:, :, 0] = img_orig[:, :, 0] * idx_fg + syn_bg[:, :, 0] * (1 - idx_fg)
        seg_img[:, :, 1] = img_orig[:, :, 1] * idx_fg + syn_bg[:, :, 1] * (1 - idx_fg)
        seg_img[:, :, 2] = img_orig[:, :, 2] * idx_fg + syn_bg[:, :, 2] * (1 - idx_fg)
        seg_img = cv2.resize(seg_img,(imgW, imgH))
        cv2.imshow('input', img_orig)

        cv2.imshow('frame', seg_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def evaluateModelPIL(model, savedir, saveloc, mean, std, imgW, imgH, videoName, Lovasz):
    # gloabl mean and std values


    syn_bg = cv2.imread(os.path.join(savedir,'syn_bg.jpg'))
    videoOut = os.path.join(saveloc,videoName)
    print("video is saved in " + videoOut)

    video = cv2.VideoWriter(videoOut, cv2.VideoWriter_fourcc(*'mp4v'), 30, (imgW, imgH))
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    success = True
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
        success =False

    while (success):
        success, img= cap.read()

        if not success or img is None:
            vidcap.release()
            break

        img_orig = np.copy(img)
        # PILImage.fromarray(img_orig).show()

        img = cv2.resize(img, (imgW, imgH))
        # PILImage.fromarray(img).show()

        img = img.astype(np.float32)
        img_tensor = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
        img_tensor = F.normalize(img_tensor, mean, std)  # normalize the tensor


        with torch.no_grad():
            img_variable = torch.autograd.Variable(img_tensor)

            if torch.cuda.is_available():
                img_variable = img_variable.cuda()

            img_variable= torch.unsqueeze(img_variable,0)
            img_out = model(img_variable)
        img_orig = cv2.resize(img_orig, (imgW, imgH))

        if Lovasz:
            classMap_numpy = (img_out[0].data.cpu() > 0).numpy()[0]
        else:
            classMap_numpy = img_out[0].max(0)[1].byte().data.cpu().numpy()

        idx_fg = (classMap_numpy == 1)

        syn_bg = cv2.resize(syn_bg, (img_out.size(3), img_out.size(2)))
        img_orig = cv2.resize(img_orig, (img_out.size(3), img_out.size(2)))

        seg_img = 0 * img_orig
        seg_img[:, :, 0] = img_orig[:, :, 0] * idx_fg + syn_bg[:, :, 0] * (1 - idx_fg)
        seg_img[:, :, 1] = img_orig[:, :, 1] * idx_fg + syn_bg[:, :, 1] * (1 - idx_fg)
        seg_img[:, :, 2] = img_orig[:, :, 2] * idx_fg + syn_bg[:, :, 2] * (1 - idx_fg)
        seg_img = cv2.resize(seg_img,(imgW, imgH))
        cv2.imshow('input', img_orig)
        cv2.imshow('frame', seg_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # cv2.destroyAllWindows()

def DemoWebcam(model, Maxfile, savedir, model_name,  videoName,h,w , mean, std, Lovasz, pil=True):
    # read all the images in the folder

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(Maxfile))
    else:
        model.load_state_dict(torch.load(Maxfile,"cpu"))

    model.eval()


    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    if not os.path.isdir(savedir+model_name):
        os.mkdir(savedir+model_name)
    saveloc = savedir+model_name
    if pil:
        evaluateModelPIL(model, savedir, saveloc, mean, std, w, h, videoName, Lovasz)
    else:
        evaluateModelCV(model, savedir, saveloc, mean, std, w, h, videoName, Lovasz)


if __name__ == '__main__':
    import models


    parser = ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default='../setting/SINet.json',
                        help='JSON file for configuration')

    Max_name = "../result/Dnc_SINet11-24_2218/model_3.pth"
    logdir= "../video/Dnc_SINet11-24_2218"
    mean = [107.304565, 115.69884, 132.35703 ]
    std = [63.97182, 65.1337, 68.29726]
    args = parser.parse_args()

    with open(args.config) as fin:
        config = json.load(fin)

    train_config = config['train_config']
    data_config = config['data_config']

    model_name = "Dnc_SINet"

    Lovasz = train_config["loss"] == "Lovasz"
    if Lovasz:
        train_config["num_classes"] = train_config["num_classes"] -1

    model = models.__dict__[model_name](classes=train_config["num_classes"],
                                        p=train_config["p"], q=train_config["q"], chnn=train_config["chnn"])

    if torch.cuda.device_count() > 0:
        model=model.cuda()

    ExportVideo(model, Max_name, "../video/", logdir, "video1.mp4", data_config["h"], data_config["w"], mean, std, Lovasz,
                pil=False)
    #
    # model_name = ["Stage2_ExtremeC3NetV2",] #  Stage2_ExtremeC3NetV2 or  ExtremeC3Net_small
    # # file_loc = [""] # directory of model file
    # weight_list=[  "./result/Stage2_ExtremeC3Net_240_320/model_288.pth"]
    # # weight_list=[  "./result/Stage2_ExtremeC3NetV208-18_1329/model_283.pth"]
    #
    # Video_name = ["video/video1.mp4", "video/video2.mp4"]
    #
    # args.model_name =model_name
    # # args.file_loc = file_loc
    # args.weight_list= weight_list
    # args.video = Video_name
    #
