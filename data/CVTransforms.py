import numpy as np
import torch
import random
import cv2
import math
from PIL import Image, ImageEnhance, ImageOps, ImageFile

set_ratio = 0.5


def aug_matrix(img_w, img_h, bbox, w, h, angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=40):
    '''
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    '''

    ratio = 1.0 * (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img_w * img_h)
    x_offset = (random.random() - 0.5) * 2 * offset
    y_offset = (random.random() - 0.5) * 2 * offset
    dx = (w - (bbox[2] + bbox[0])) / 2.0
    dy = (h - (bbox[3] + bbox[1])) / 2.0

    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0, 1.0]])

    angle = random.random() * (angle_range[1] - angle_range[0]) + angle_range[0]
    scale = random.random() * (scale_range[1] - scale_range[0]) + scale_range[0]
    scale *= np.mean([float(w) / (bbox[2] - bbox[0]), float(h) / (bbox[3] - bbox[1])])
    alpha = scale * math.cos(angle / 180.0 * math.pi)
    beta = scale * math.sin(angle / 180.0 * math.pi)

    centerx = w / 2.0 + x_offset
    centery = h / 2.0 + y_offset
    H = np.array([[alpha, beta, (1 - alpha) * centerx - beta * centery],
                  [-beta, alpha, beta * centerx + (1 - alpha) * centery],
                  [0, 0, 1.0]])

    H = H.dot(matrix_trans)[0:2, :]
    return H


def data_aug_flip(image, mask):
    if random.random() < set_ratio:
        return image, mask, False
    return image[:, ::-1, :], mask[:, ::-1], True


class Translation(object):
    def __init__(self, wi, he, padding_color=128):
        self.input_width = wi
        self.input_height = he
        self.padding_color = padding_color

    def __call__(self, img, label):
        height, width, channel = img.shape
        bbox = [0, 0, width - 1, height - 1]

        H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height / 4)

        img_aug = cv2.warpAffine(np.uint8(img), H, (self.input_width, self.input_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(self.padding_color, self.padding_color, self.padding_color))
        mask_aug = cv2.warpAffine(np.uint8(label), H, (self.input_width, self.input_height),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        img_aug_ori, mask_aug_ori, _ = data_aug_flip(img_aug, mask_aug)

        return [img_aug_ori, mask_aug_ori]


class data_aug_light(object):

    def __call__(self, image, label):
        value = random.randint(-30, 30)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = np.array(hsv_image, dtype=np.float32)
        hsv_image[:, :, 2] += value
        hsv_image[hsv_image > 255] = 255
        hsv_image[hsv_image < 0] = 0
        hsv_image = np.array(hsv_image, dtype=np.uint8)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return image, label

class texture_aug(object):
    def __call__(self, image, label):


        #aug blur
        if random.random() > set_ratio:
            select = random.random()
            if select < 0.3:
                kernalsize = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (kernalsize, kernalsize), 0)
            elif select < 0.6:
                kernalsize = random.choice([3, 5])
                image = cv2.medianBlur(image, kernalsize)
            else:
                kernalsize = random.choice([3, 5])
                image = cv2.blur(image, (kernalsize, kernalsize))

        # aug noise
        if random.random() > set_ratio:
            mu = 0
            sigma = random.random() * 10.0
            image = np.array(image, dtype=np.float32)
            image += np.random.normal(mu, sigma, image.shape)
            image[image > 255] = 255
            image[image < 0] = 0

        # aug_color
        if random.random() > set_ratio:

            random_factor = np.random.randint(4, 17) / 10.
            color_image = ImageEnhance.Color(image).enhance(random_factor)
            random_factor = np.random.randint(4, 17) / 10.
            brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
            random_factor = np.random.randint(6, 15) / 10.
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
            random_factor = np.random.randint(8, 13) / 10.
            image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

        return np.array(image), label

class data_aug_blur(object):

    def __call__(self, image, label):
        if random.random() < set_ratio:
            return image, label

        select = random.random()
        if select < 0.3:
            kernalsize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernalsize, kernalsize), 0)
        elif select < 0.6:
            kernalsize = random.choice([3, 5])
            image = cv2.medianBlur(image, kernalsize)
        else:
            kernalsize = random.choice([3, 5])
            image = cv2.blur(image, (kernalsize, kernalsize))
        return image, label


class data_aug_noise(object):

    def __call__(self, image, label):
        if random.random() < set_ratio:
            return image, label
        mu = 0
        sigma = random.random() * 10.0
        image = np.array(image, dtype=np.float32)
        image += np.random.normal(mu, sigma, image.shape)
        image[image > 255] = 255
        image[image < 0] = 0

        return image, label


class data_aug_color(object):

    def __call__(self, image, label):
        if random.random() < set_ratio:
            return image, label
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        random_factor = np.random.randint(4, 17) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(4, 17) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(6, 15) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(8, 13) / 10.
        image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        return np.uint8(np.array(image)[:,:,::-1]), label


class Scale(object):
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """
    def __init__(self, wi, he):
        '''

        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        #bilinear interpolation for RGB image
        img = cv2.resize(img, (self.w, self.h))
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return [img, label]



class RandomCropResize(object):
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between o and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = label[y1:h-y1, x1:w-x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w,h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop
        else:
            return [img, label]

class RandomCrop(object):
    '''
    This class if for random cropping
    '''
    def __init__(self, cropArea):
        '''
        :param cropArea: amount of cropping (in pixels)
        '''
        self.crop = cropArea

    def __call__(self, img, label):

        if random.random() < 0.5:
            h, w = img.shape[:2]
            img_crop = img[self.crop:h-self.crop, self.crop:w-self.crop]
            label_crop = label[self.crop:h-self.crop, self.crop:w-self.crop]
            return img_crop, label_crop
        else:
            return [img, label]



class RandomFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
            x1 = 0#random.randint(0, 1) #if you want to do vertical flip, uncomment this line
            if x1 == 0:
                image = cv2.flip(image, 0) # horizontal flip
                label = cv2.flip(label, 0) # horizontal flip
            else:
                image = cv2.flip(image, 1) # veritcal flip
                label = cv2.flip(label, 1)  # veritcal flip
        return [image, label]


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32)
        for i in range(3):
            image[:,:,i] -= self.mean[i]
        for i in range(3):
            image[:,:, i] /= self.std[i]

        return [image, label]

class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''
    def __init__(self, scale=1):
        '''
        :param scale: ESPNet-C's output is 1/8th of original image size, so set this parameter accordingly
        '''
        self.scale = scale # original images are 2048 x 1024

    def __call__(self, image, label):

        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)), interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2,0,1))

        image_tensor = torch.from_numpy(image).div(255)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int)).div(255) #torch.from_numpy(label)

        return [image_tensor, label_tensor]

class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
