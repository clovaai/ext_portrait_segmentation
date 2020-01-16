import numpy as np
import torch
import random
import math
import numbers
from torchvision.transforms import Pad
from torchvision.transforms import functional as F
from PIL import Image
from .CVTransforms import ImageEnhance
import cv2

class data_aug_color(object):

    def __call__(self, image, label):
        if random.random() < 0.5:
            return image, label
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        random_factor = np.random.randint(4, 17) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(4, 17) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(6, 15) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(8, 13) / 10.
        image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        return image, label


class Normalize(object):
    '''
        Normalize the tensors
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scaleIn=1):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''

        self.mean = mean
        self.std = std
        self.scale = scaleIn


    def __call__(self, rgb_img, label_img=None):

        if self.scale != 1:
            w, h = label_img.size
            label_img = label_img.resize((w//self.scale, h//self.scale), Image.NEAREST)


        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, self.mean, self.std) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))


        return rgb_img, label_img


class DoubleNormalize(object):
    '''
        Normalize the tensors
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale1=0.5, scale2=1):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''

        self.mean = mean
        self.std = std
        self.scale1 = scale1
        self.scale2 = scale2


    def __call__(self, rgb_img, label_img=None):

        label1 = label_img
        label2 = label_img
        if self.scale1 != 1:
            w, h = label_img.size
            label1 = label1.resize((w//self.scale1, h//self.scale1), Image.NEAREST)

        if self.scale2 != 1:
            w, h = label_img.size
            label2 = label2.resize((w//self.scale2, h//self.scale2), Image.NEAREST)

        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, self.mean, self.std) # normalize the tensor
        label1 = torch.LongTensor(np.array(label1).astype(np.int64))
        label2 = torch.LongTensor(np.array(label2).astype(np.int64))


        return rgb_img, label1, label2




class RandomFlip(object):
    '''
        Random Flipping
    '''
    def __call__(self, rgb_img, label_img):
        if random.random() < 0.5:
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb_img, label_img


class RandomScale(object):
    '''
    Random scale, where scale is logrithmic
    '''
    def __init__(self, scale=(0.5, 1.0)):
        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size
        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        rgb_img = rgb_img.resize(new_size, Image.ANTIALIAS)
        label_img = label_img.resize(new_size, Image.NEAREST)
        return rgb_img, label_img


class RandomCrop(object):
    '''
    Randomly crop the image
    '''
    def __init__(self, crop_size, ignore_idx=255):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.ignore_idx = ignore_idx

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size
        pad_along_w = max(0, int((1 + self.crop_size[0] - w) / 2))
        pad_along_h = max(0, int((1 + self.crop_size[1] - h) / 2))
        # padd the images
        rgb_img = Pad(padding=(pad_along_w, pad_along_h), fill=0, padding_mode='constant')(rgb_img)
        label_img = Pad(padding=(pad_along_w, pad_along_h), fill=self.ignore_idx, padding_mode='constant')(label_img)

        i, j, h, w = self.get_params(rgb_img, self.crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)
        return rgb_img, label_img

class Resize(object):
    '''
        Resize the images
    '''
    def __init__(self, size=(512, 512)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, rgb_img, label_img):
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        label_img = label_img.resize(self.size, Image.NEAREST)
        return rgb_img, label_img


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
