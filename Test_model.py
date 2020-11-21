'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from etc.Tensor_logger import Logger
from data.dataloader import get_dataloader
import models
from etc.help_function import *
from etc.utils import *
from etc.Visualize_video import ExportVideo
from etc.Visualize_webCam import DemoWebcam
from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./setting/Test_SINet.json', help='JSON file for configuration')
    parser.add_argument('-n', '--use_nsml', type=bool, default=False, help='Play with NSML!')
    parser.add_argument('-d', '--decoder_only', type=bool, default=False, help='Decoder only training')
    parser.add_argument('-o', '--optim', type=str, default="Adam", help='Adam , SGD, RMS')
    parser.add_argument('-s', '--lrsch', type=str, default="multistep", help='step, poly, multistep, warmpoly')
    parser.add_argument('-t', '--wd_tfmode', type=bool, default=True, help='Play with NSML!')
    parser.add_argument('-w', '--weight_decay', type=float, default=2e-4, help='value for weight decay')
    parser.add_argument('-v', '--visualize', type=bool, default=True, help='visualize result image')
    parser.add_argument('--demoWebcam', type=bool, default=True, help='visualize result image')
    parser.add_argument('--demoVideo', type=str, default="video1.mp4", help='visualize result image')

    args = parser.parse_args()
    others= args.weight_decay*0.05
    ############### setting framework ##########################################
    with open(args.config) as fin:
        config = json.load(fin)
    test_config = config['test_config']
    data_config = config['data_config']

    args.optim = test_config["optim"]
    args.lrsch = test_config["lrsch"]
    args.wd_tfmode = test_config["wd_tfmode"]
    args.weight_decay = test_config["weight_decay"]
    others = args.weight_decay * 0.05

    if test_config["loss"] == "Lovasz":
        test_config["num_classes"] = 1
        print("Use Lovasz loss ")
        Lovasz = True

    else:
        print("Use Cross Entropy loss ")
        Lovasz = False

    if not os.path.isdir(test_config['save_dir']):
        os.mkdir(test_config['save_dir'])

    print("Run : " + test_config["Model"])
    D_ratio = []
    if test_config["Model"].startswith('Stage1'):
        model = models.__dict__[test_config["Model"]](
            p=test_config["p"], q=test_config["q"], classes=test_config["num_classes"])

    elif test_config["Model"].startswith('Stage2'):
        model = models.__dict__[test_config["Model"]](classes=test_config["num_classes"],
                                                       p=test_config["p"], q=test_config["q"])

    elif test_config["Model"].startswith('ExtremeC3Net_small'):
        model = models.__dict__[test_config["Model"]](classes=test_config["num_classes"],
                                                       p=test_config["p"], q=test_config["q"])
    elif test_config["Model"].startswith('Dnc_SINet'):
        model = models.__dict__[test_config["Model"]](
            classes=test_config["num_classes"], p=test_config["p"], q=test_config["q"],
            chnn=test_config["chnn"])

    model_name = test_config["Model"]

    print(test_config["num_classes"])
    batch = torch.FloatTensor(1, 3, data_config["w"], data_config["h"])
    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(batch)
    N_flop = model.compute_average_flops_cost()
    total_paramters = netParams(model)

    color_transform = Colorize(test_config["num_classes"])

    #################### common model setting and opt setting  #######################################

    if args.use_nsml:
        from nsml import DATASET_PATH
        data_config['data_dir'] = os.path.join(DATASET_PATH, 'train')
    
    Max_name = test_config["weight_name"]
    if torch.cuda.device_count() > 0:
        model.load_state_dict(torch.load(Max_name))
    else:
        model.load_state_dict(torch.load(Max_name, "cpu"))

    use_cuda = torch.cuda.is_available()
    num_gpu = torch.cuda.device_count()

    if use_cuda:
        print("Use gpu : %d" % torch.cuda.device_count())
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
            print("make DataParallel")
        model = model.cuda()
        print("Done")

    ###################################stage Enc setting ##############################################
    if (not args.decoder_only):
        logger, this_savedir = info_setting(test_config['save_dir'], test_config["Model"], total_paramters, N_flop)
        logger.flush()
        logdir = this_savedir.split(test_config['save_dir'])[1]
        my_logger = Logger(8097, './logs/' + logdir, args.use_nsml)

        trainLoader,  valLoader, data = get_dataloader(data_config)

        print(data['mean'])
        print(data['std'])
        weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
        print(weight)

        if test_config["loss"] == "Lovasz":
            from etc.lovasz_losses import lovasz_hinge
            criteria = lovasz_hinge(ignore=data_config["ignore_idx"])
        else:
            from etc.Criteria import CrossEntropyLoss2d
            criteria = CrossEntropyLoss2d(weight,ignore=data_config["ignore_idx"])  # weight

        if num_gpu > 0:
            weight = weight.cuda()
            criteria = criteria.cuda()

        print("init_lr: " + str(test_config["learning_rate"]) + "   batch_size : " + str(data_config["batch_size"]) +
              args.lrsch + " sch use weight and class " + str(test_config["num_classes"]))
        print("logs saved in " + logdir + "\tlr sch: " + args.lrsch + "\toptim method: " + args.optim +
              "\ttf style : " + str(args.wd_tfmode) + "\tbn-weight : " + str(others))

        print('Flops:  {}'.format(flops_to_string(N_flop)))
        print('Params: ' + get_model_parameters_number(model))
        print('Output shape: {}'.format(list(out.shape)))
        print(total_paramters)

    ################################ start Enc train ##########################################
        if args.visualize:
            lossVal, ElossVal, mIOU_val, save_input, save_est, save_gt = \
                val_edge(num_gpu, valLoader, model, criteria, Lovasz, args.visualize)
            if test_config["loss"] == "Lovasz":
                grid_outputs = torchvision.utils.make_grid(color_transform((save_est[0] > 0).cpu().data), nrow=6)
            else:
                grid_outputs = torchvision.utils.make_grid(
                    color_transform(save_est[0].unsqueeze(0).cpu().max(1)[1].data), nrow=6)
            my_logger.image_summary(torchvision.utils.make_grid(save_input[0], normalize=True),
                                    opts=dict(title=f'VAL img (epoch: {0})', caption=f'VAL img (epoch: {0})'))

            my_logger.image_summary(grid_outputs,
                                    opts=dict(title=f'VAL output (epoch: {0}, step: {str(mIOU_val)})',
                                              caption=f'VAL output (epoch: {0}, step: {str(mIOU_val)})', ))

        else:
            lossVal, ElossVal, mIOU_val = val_edge(num_gpu, valLoader, model, criteria, Lovasz)

        print("mIOU(val) = %.4f" %mIOU_val)
    print("========== TEST FINISHED ===========")
    mean = data['mean']
    std = data['std']
    print(mean)
    print(std)
    if data_config["dataset_name"] =="pilportrait":
        isPILlodear=True
    else:
        isPILlodear=False

    if args.demoWebcam:
        DemoWebcam(model, Max_name, "./video", logdir, "videoCam.mp4", data_config["h"], data_config["w"], mean, std, Lovasz,
                    pil=isPILlodear)

    if args.demoVideo !="":
        ExportVideo(model, Max_name, "./video", logdir, args.demoVideo, data_config["h"], data_config["w"], mean, std, Lovasz,
                pil=isPILlodear)










