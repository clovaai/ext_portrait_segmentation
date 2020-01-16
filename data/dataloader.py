import os
import data.CVTransforms as cvTransforms
import data.PILTransform as pilTransforms

from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms
import data.DataSet as myDataLoader
import torch
import data.loadData as ld
import pickle



def portrait_CVdataloader(cached_data_file, data_dir, classes, batch_size, scaleIn,
                           w=180, h=320, edge=False, num_work=4, Enc = True, Augset= True):

    if not os.path.isfile(cached_data_file):
        if Augset:
            additional_data = []
            additional_data.append('/Nukki/baidu_V1/')
            additional_data.append('/Nukki/baidu_V2/')

            dataLoad = ld.LoadData(data_dir, classes, cached_data_file, additional=additional_data)
            data = dataLoad.processDataAug()
        else:
            dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
            data = dataLoad.processData()

        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))

    trainDataset_main = cvTransforms.Compose([
        cvTransforms.Translation(w,h),
        # cvTransforms.data_aug_light(),
        cvTransforms.data_aug_color(),
        cvTransforms.data_aug_blur(),
        cvTransforms.data_aug_noise(),
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(w, h),
        cvTransforms.ToTensor(scaleIn),
        #
    ])



    valDataset = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(w, h),
        cvTransforms.ToTensor(scaleIn),
        #
    ])


    print("This stage is Enc" +str(Enc))
    print(" Load public Baidu train dataset")
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.CVDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main, edge= edge, Enc=Enc),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    print(" Load public val dataset")

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.CVDataset(data['valIm'], data['valAnnot'], transform=valDataset, edge= True, Enc=Enc),
        batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)


    return trainLoader, valLoader, data


def portrait_multiCVdataloader(cached_data_file, data_dir, classes, batch_size, scaleIn,
                           w=180, h=320, edge=False, num_work=4, Enc = True, Augset= True):

    if not os.path.isfile(cached_data_file):
        if Augset:
            additional_data = []

            additional_data.append('/Nukki/baidu_V1/')
            additional_data.append('/Nukki/baidu_V2/')

            dataLoad = ld.LoadData(data_dir, classes, cached_data_file, additional=additional_data)
            data = dataLoad.processDataAug()
        else:
            dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
            data = dataLoad.processData()

        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))


    trainDataset_main = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(w, h),
        cvTransforms.RandomCropResize(32),
        cvTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64).
        cvTransforms.ToTensor(scaleIn),
        #
    ])

    trainDataset_main2 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(224,224),
        cvTransforms.RandomCropResize(16),
        cvTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64).
        cvTransforms.ToTensor(scaleIn),
        #
    ])


    trainDataset_main3 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(int(w * 0.8), int(h * 0.8)),
        cvTransforms.RandomCropResize(24),
        cvTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64).
        cvTransforms.ToTensor(scaleIn),
        #
    ])
    valDataset = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(224, 224),
        cvTransforms.ToTensor(scaleIn),
        #
    ])


    print("This stage is Enc" +str(Enc))
    print(" Load public Baidu train dataset")
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.CVDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main, edge= edge, Enc=Enc),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)


    trainLoader2 = torch.utils.data.DataLoader(
        myDataLoader.CVDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main2, edge= edge, Enc=Enc),
        batch_size=int(1.5*batch_size), shuffle=True, num_workers=num_work, pin_memory=True)


    trainLoader3 = torch.utils.data.DataLoader(
        myDataLoader.CVDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main3, edge= edge, Enc=Enc),
        batch_size=int(1.8*batch_size), shuffle=True, num_workers=num_work, pin_memory=True)

    print(" Load public val dataset")

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.CVDataset(data['valIm'], data['valAnnot'], transform=valDataset, edge= True, Enc=Enc),
        batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)



    return trainLoader, trainLoader2, trainLoader3, valLoader, data


def portraitPIL_Doublerandscalecrop(cached_data_file, data_dir, classes, batch_size, scale=(0.8, 1.0),
                        size=(1024, 512), scale1=1, scale2=2, ignore_idx=255, edge=False ,num_work=6,
                         Augset= True):

    print("This input size is  " +str(size))

    if not os.path.isfile(cached_data_file):
        if Augset:
            additional_data = []

            additional_data.append('/Nukki/baidu_V1/')
            additional_data.append('/Nukki/baidu_V2/')

            dataLoad = ld.LoadData(data_dir, classes, cached_data_file, additional=additional_data)
            data = dataLoad.processDataAug()
        else:
            dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
            data = dataLoad.processData()

        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))


    if isinstance(size, tuple):
        size = size
    else:
        size = (size, size)

    if isinstance(scale, tuple):
        scale = scale
    else:
        scale = (scale, scale)


    train_transforms = pilTransforms.Compose(
        [
            # pilTransforms.data_aug_color(),
            pilTransforms.RandomScale(scale=scale),
            pilTransforms.RandomCrop(crop_size=size,ignore_idx=ignore_idx),
            pilTransforms.RandomFlip(),
            pilTransforms.DoubleNormalize(scale1=scale1, scale2=scale2)
        ]
    )
    val_transforms = pilTransforms.Compose(
        [
            pilTransforms.Resize(size=size),
            # pilTransforms.RandomScale(scale=scale),
            # pilTransforms.RandomCrop(crop_size=size, ignore_idx=ignore_idx),
            # pilTransforms.RandomFlip(),
            pilTransforms.DoubleNormalize(scale1=scale2, scale2=1)
        ]
    )
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['trainIm'], data['trainAnnot'], Double=True,
                                ignore_idx=ignore_idx, edge=edge, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['valIm'], data['valAnnot'], Double=True,
                                ignore_idx=ignore_idx, edge=True, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)

    return trainLoader, valLoader, data

def get_dataloader(args): #cash, dataset_name, data_dir, classes, batch_size, scaleIn=1, w=180, h=320, Edge= False):

    dataset_name = args["dataset_name"]
    data_file = args["cash"]
    data_dir = args["data_dir"]
    classes = args["classes"]
    batch_size = args["batch_size"]
    w = args["w"]
    h = args["h"]
    Edge = args["Edge"]
    num_work = args["num_work"]
    Aug = args["Aug_dataset"]


    if dataset_name=='CVportrait':
        scaleIn = args["scaleIn"]
        Enc = args["Enc"]
        print(" This data load w = %d h = %d scaleIn = %d" % (w, h, scaleIn))
        return portrait_CVdataloader(data_file, data_dir, classes, batch_size, scaleIn,
                                      w=w, h=h, edge= Edge,num_work=num_work, Augset= Aug, Enc=Enc)

    elif dataset_name=="CVmultiportrait":
        scaleIn = args["scaleIn"]
        Enc = args["Enc"]
        print(" This data load w = %d h = %d scaleIn = %d" % (w, h, scaleIn))
        return portrait_multiCVdataloader(data_file, data_dir, classes, batch_size, scaleIn,
                                     w=w, h=h, edge=Edge, num_work=num_work, Augset=Aug, Enc=Enc)


    elif dataset_name =='pilportrait':

        trainsize = (w,h)
        scale1 = args["scale1"]
        scale2 = args["scale2"]
        print(" This data load w = %d h = %d scaleIn = %d" % (w, h, scale1))
        return portraitPIL_Doublerandscalecrop(data_file, data_dir, classes, batch_size,
                                size=trainsize, scale1=scale1, scale2=scale2, edge= Edge, Augset= Aug)

    else:
        print(dataset_name + "is not implemented")
        raise NotImplementedError


if __name__ == '__main__':
    portrait_CVdataloader("None.p", "../../Link512DATA/", 2, 60, 1, w=224, h=224, edge=False)