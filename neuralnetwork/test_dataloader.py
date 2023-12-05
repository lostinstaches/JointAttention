import torch
from torch import nn

import json

import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from scipy import io as sp_io

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from copy import deepcopy
from scipy import io as sp_io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import torchvision.transforms as transforms
from PIL import Image
image_size = 224
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
class TestDataset(Dataset):
    #def __init__(self, image_folder, gaze_folder, label_folder):
    #    self.image_folder = image_folder
    #    self.gaze_folder = gaze_folder
    #    self.label_folder = label_folder

    #   self.image_files = sorted(os.listdir(image_folder))
    #  self.gaze_files = sorted(os.listdir(gaze_folder))
    # self.label_files = sorted(os.listdir(label_folder))
    def __init__(self, path, part='test', cut=np.inf, split = 'random'):
        self.part = part
        self.data_x = None
        self.data_y = None
        
        part == 'test'
        self.create_test()
        self.read_data('test')

    def norm_imm(self, im):
        #print(im.shape)
        for j in range(0,len(im)):
            data = Image.fromarray(im[j])
            im[j]= tfms(data)
        #print(im.shape)
        # import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        t=np.stack(im)
        return t

    def create_test(self):
        
        folder = 'C:\\Users\\laura\\Documents\\Dados\\Arianna_Alessandro\\p1'
        

        
        gaze = pd.read_csv(folder + '/selected_gaze_ch.txt', sep=" ", header=None)
        gaze_df = np.asarray(gaze).astype('float32')
        hh = np.arange(6,10)
        gaze_df = np.delete(gaze_df,hh,1)

        labels = pd.read_csv(folder + '/labels_ch.txt', sep=" ", header=None)
        labels_fin= np.array(labels)

        

        images_in = open(folder + '/IMAGE.npy','rb')
        images = np.load(images_in, allow_pickle=True)
        image_fin = self.norm_imm(images)

        data = gaze_df, labels_fin, image_fin
        with open('D:\\Pickles_attention\\no_hbb_test.pkl', 'wb') as f:
            pickle.dump(data, f)




    #def __getitem__(self, index):
    #    image_path = os.path.join(self.image_folder, self.image_files[index])
     #   images_in = open(image_path + '/IMAGESIMAGE.npy','rb')
      #  images = np.load(images_in, allow_pickle=True)

       # gaze = pd.read_csv(gaze_path + '/GAZE_ELISA.csv', sep=" ", header=None)
        #gaze_df = np.asarray(gaze).astype('float32')

        #gaze_path = os.path.join(self.gaze_folder, self.gaze_files[index])
        #gaze = pd.read_csv(gaze_path + '/GAZE_ELISA.csv', sep=" ", header=None)
        #gaze_df = np.asarray(gaze).astype('float32')
        

        #label_path = os.path.join(self.label_folder, self.label_files[index])
        #labels = pd.read_csv(label_path + '/labels_elisa.csv', sep=" ", header=None)
        #labels_fin = np.array(labels)

        #return images, gaze_df, labels_fin

    def read_data(self, label):

        gaze_df, labels_fin, images = pickle.load(open('D:\\Pickles_attention\\no_hbb_test.pkl', 'rb'), encoding='latin1')
        #import pdb; pdb.set_trace()
        if label == 'test':
            self.gaze = gaze_df
            self.immagine = images
            self.output = labels_fin

    def __len__(self):
        return len(self.gaze)
    
    def __getitem__(self, idx):
        gaze = self.gaze[idx]
        output = self.output[idx]
        immagine = self.immagine[idx]
        return torch.tensor(gaze).float(), torch.tensor(immagine), torch.tensor(output).float()

