import numpy as np
import cv2
import pickle
import os

class LoadData:
    '''
    Class to laod the data
    '''
    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10, additional=None):
        '''
        :param data_dir: directory where the dataset is kept
        :param classes: number of classes in the dataset
        :param cached_data_file: location where cached file has to be stored
        :param normVal: normalization value, as defined in ERFNet paper
        '''
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.addvalImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.addvalAnnotList = list()
        self.cached_data_file = cached_data_file
        self.train_txt=list()
        self.val_txt=list()
        self.additional = additional

    def compute_class_weights(self, histogram):
        '''
        Helper function to compute the class weights
        :param histogram: distribution of class samples
        :return: None, but updates the classWeights variable
        '''
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False, addtional=None):
        '''
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        '''
        if trainStg == True:
            global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(self.data_dir + '/Portrait/' + fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                # line_arr = line.split(',')
                name_num = int(line)
                img_file = ((self.data_dir).strip() + '/Portrait/images_data_crop/' + str(name_num).zfill(5) +'.jpg')
                label_file = ((self.data_dir).strip() + '/Portrait/GT_png/' + str(name_num).zfill(5) + '_mask.png')

                # print(label_file)
                label_img = cv2.imread(label_file, 0)
                label_img = label_img/255
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if trainStg == True:
                    hist = np.histogram(label_img, self.classes)
                    global_hist += hist[0]
                    try:
                        rgb_img = cv2.imread(img_file)
                        self.mean[0] += np.mean(rgb_img[:, :, 0])
                        self.mean[1] += np.mean(rgb_img[:, :, 1])
                        self.mean[2] += np.mean(rgb_img[:, :, 2])

                        self.std[0] += np.std(rgb_img[:, :, 0])
                        self.std[1] += np.std(rgb_img[:, :, 1])
                        self.std[2] += np.std(rgb_img[:, :, 2])

                        self.trainImList.append(img_file)
                        self.trainAnnotList.append(label_file)
                        no_files += 1
                        self.train_txt.append(str(name_num).zfill(5))
                    except:
                        print("Train has problem" + img_file)
                else:

                    rgb_img = cv2.imread(img_file)
                    try:
                        if len(rgb_img.shape) >2:
                            self.valImList.append(img_file)
                            self.valAnnotList.append(label_file)
                            self.val_txt.append(str(name_num).zfill(5))

                        else:
                            print("Val has problem" + img_file)
                    except:
                        print("Val has problem" + img_file)

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check.')
                    print('Label Image ID: ' + label_file)

     ############ add additional dataset with ##################################

        if addtional !=None:
            for i in range(len(addtional)):
                this_additoinal = addtional[i]
                print(this_additoinal)
                with open(self.data_dir + this_additoinal + fileName, 'r') as textFile:
                    for line in textFile:
                        # we expect the text file to contain the data in following format
                        # <RGB Image>, <Label Image>
                        # line_arr = line.split(',')
                        img_file = ((self.data_dir).strip() + this_additoinal+'input/' + line.strip())
                        label_file = ((self.data_dir).strip() + this_additoinal+'target/' + line.strip())
                        label_img = cv2.imread(label_file, 0)
                        if os.path.isfile(label_file) == True:

                            label_bool = 255*((label_img >200).astype(np.uint8))
                            label_img = label_bool / 255
                            unique_values = np.unique(label_img)
                            max_val = max(unique_values)
                            min_val = min(unique_values)

                            max_val_al = max(max_val, max_val_al)
                            min_val_al = min(min_val, min_val_al)

                            if trainStg == True:
                                hist = np.histogram(label_img, self.classes)
                                global_hist += hist[0]
                                try:
                                    rgb_img = cv2.imread(img_file)
                                    self.mean[0] += np.mean(rgb_img[:, :, 0])
                                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                                    self.std[0] += np.std(rgb_img[:, :, 0])
                                    self.std[1] += np.std(rgb_img[:, :, 1])
                                    self.std[2] += np.std(rgb_img[:, :, 2])

                                    self.trainImList.append(img_file)
                                    self.trainAnnotList.append(label_file)
                                    no_files += 1

                                except:
                                    print("Train has problem" + img_file)
                            else:

                                rgb_img = cv2.imread(img_file)
                                try:
                                    if len(rgb_img.shape) > 2:
                                        self.addvalImList.append(img_file)
                                        self.addvalAnnotList.append(label_file)
                                    else:
                                        print("add Val has problem" + img_file)
                                except:
                                    print("add Val has problem" + img_file)

                            if max_val > (self.classes - 1) or min_val < 0:
                                print('Labels can take value between 0 and number of classes.')
                                print('Some problem with labels. Please check.')
                                print('Label Image ID: ' + label_file)
                        else:
                            print(label_file)

        if trainStg == True:
            # divide the mean and std values by the sample space size
            self.mean /= no_files
            self.std /= no_files

            #compute the class imbalance information
            self.compute_class_weights(global_hist)
        return 0

    def processDataAug(self):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        print('Processing training data')

        return_val1 = self.readFile('train.txt', True, addtional=self.additional)

        print('Processing validation data')
        return_val2 = self.readFile('val.txt', addtional=self.additional )

        print('Pickling data')
        if (return_val1 ==0 and return_val2 ==0 ):
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList
            data_dict['addvalIm'] = self.addvalImList
            data_dict['addvalAnnot'] = self.addvalAnnotList
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            if not os.path.isdir("./pickle_file"):
                os.mkdir("./pickle_file")
            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        else:
            print("There is problem")
            exit(0)
            return None

    def processData(self):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        print('Processing training data')

        return_val1 = self.readFile('train.txt', True)

        print('Processing validation data')
        return_val2 = self.readFile('val.txt')

        print('Pickling data')
        if (return_val1 == 0 and return_val2 == 0):
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList

            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            if not os.path.isdir("./pickle_file"):
                os.mkdir("./pickle_file")
            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            # with open('EG1800_train.txt', 'w') as f:
            #     for item in self.train_txt:
            #         f.write("%s\n" % item)
            # with open('EG1800_val.txt', 'w') as f:
            #     for item in self.val_txt:
            #         f.write("%s\n" % item)
            return data_dict
        else:
            print("There is problem")
            exit(0)
            return None





