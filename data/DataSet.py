import torch
import cv2
import torch.utils.data
import numpy as np
from PIL import Image



class CVDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, Enc=True, transform=None, edge=False):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe CVTransforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform
        print("This num of data is " +str(len(imList)))
        self.edge = edge
        if Enc :
            self.kernel_size = 5
        else:
            self.kernel_size = 15

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        label_bool = 255 * ((label > 200).astype(np.uint8))

        if self.transform:
            [image, label] = self.transform(image, label_bool)
        if self.edge:
            np_label = 255 * label.data.numpy().astype(np.uint8)
            kernel = np.ones((self.kernel_size , self.kernel_size ), np.uint8)
            erosion = cv2.erode(np_label, kernel, iterations=1)
            dilation = cv2.dilate(np_label, kernel, iterations=1)
            boundary = dilation - erosion
            edgemap = 255 * torch.ones_like(label)
            edgemap[torch.from_numpy(boundary) > 0] = label[torch.from_numpy(boundary) > 0]
            return (image, label, edgemap)
        else:
            return (image, label)




class PILDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, transform=None, Double=False, ignore_idx=255, edge=True):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe CVTransforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform
        self.Double = Double
        self.ignore_idx = ignore_idx
        self.edge = edge

    def __len__(self):
        return len(self.imList)

    def Make_boundary(self, label, k_size):
        np_label = label.data.numpy().astype(np.uint8)
        target_label = 255 * np_label * (np_label == 1).astype(np.uint8)
        ignore_label = np_label * (np_label == self.ignore_idx).astype(np.uint8)


        kernel = np.ones((k_size, k_size), np.uint8)
        erosion = cv2.erode(target_label, kernel, iterations=1)
        dilation = cv2.dilate(target_label, kernel, iterations=1)
        boundary = dilation - erosion
        edgemap = 255 * torch.ones_like(label)
        edgemap[torch.from_numpy(boundary) > 0] = label[torch.from_numpy(boundary) > 0]
        edgemap[torch.from_numpy(ignore_label)>0] = self.ignore_idx

        return edgemap



    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = Image.open(image_name).convert('RGB')

        label = (np.array(Image.open(label_name)) > 200).astype(np.uint8)
        label = Image.fromarray(label, mode="L")

        if self.Double:
            if self.transform:
                [image, label_coarse, label] = self.transform(image, label)

            if self.edge:
                edgemap = self.Make_boundary(label,15)
                return (image, label_coarse, label, edgemap)

            else:
                return (image, label_coarse, label)
        else:
            if self.transform:
                [image, label] = self.transform(image, label)
            if self.edge:
                edgemap = self.Make_boundary(label,15)
                return (image, label, edgemap)

            else:
                return (image, label)

if __name__ == '__main__':
    import cv2
    import numpy as np
    from PIL import Image

    label_name = "../../../Link512DATA/Nukki/baidu_V1/target/1.png"

    label = (np.array(Image.open(label_name)) > 200).astype(np.uint8)
    label[0:50,300:] = 255
    Img_label= 100*label
    Img_label[0:50,300:] = 255

    label = torch.from_numpy(label)

    ignore_idx=255
    k_size= 15

    np_label = label.data.numpy().astype(np.uint8)
    target_label = 255 * np_label*(np_label == 1).astype(np.uint8)
    ignore_label = np_label*(np_label == ignore_idx).astype(np.uint8)

    kernel = np.ones((k_size, k_size), np.uint8)
    erosion = cv2.erode(target_label, kernel, iterations=1)
    dilation = cv2.dilate(target_label, kernel, iterations=1)
    boundary = dilation - erosion
    edgemap = 2 * torch.ones_like(label)
    edgemap[torch.from_numpy(boundary) > 0] = 100*label[torch.from_numpy(boundary) > 0]
    edgemap[torch.from_numpy(ignore_label) > 0] = ignore_idx

    Image.fromarray(Img_label).show()
    Image.fromarray(edgemap.data.numpy()).show()

    # label = cv2.imread(label_name, 0)
    # label_bool = 255 * ((label > 200).astype(np.uint8))
    # label_tensor = torch.LongTensor(np.array(label_bool, dtype=np.int)).div(255)  # torch.from_numpy(label)
    # cv2.imshow("label", label)
    #
    # np_label = 255 * label_tensor.data.numpy().astype(np.uint8)
    # kernel = np.ones((21, 21), np.uint8)
    # erosion = cv2.erode(np_label, kernel, iterations=1)
    # dilation = cv2.dilate(np_label, kernel, iterations=1)
    # boundary = dilation-erosion
    # cv2.imshow("boundary",boundary)
    # edgemap = 255*torch.ones_like(label_tensor)
    # edgemap[torch.from_numpy(boundary)>0]=100*label_tensor[torch.from_numpy(boundary)>0]
    # cv2.imshow("Edge", edgemap.data.numpy().astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

