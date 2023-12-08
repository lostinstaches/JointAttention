import torch
from torch import nn

import json

import os
import torch
from torch.utils.data import Dataset
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

class AttentionDataset(Dataset):
    # TODO: Missing data corruption
    def __init__(self, path, part='train', cut=np.inf, split = 'random'):
        self.part = part
        self.data_x = None
        self.data_y = None
        if part == 'train':
            # self.create_train_test_split(split)
            self.read_data('train')
        elif part == 'test':
            self.read_data('test')
        elif part == 'val':
            self.read_data('val')
    def _convert_one_hot_entry(self, array, idx):
        first_part = array[:idx]
        action = int(array[idx])
        second_part = array[idx+1:]

        one_hot_vector = np.zeros(4)
        one_hot_vector[action-1] = 1

        transformed_vector = np.concatenate([first_part, one_hot_vector, second_part], axis=0)

        return transformed_vector

    def norm_imm(self, im):
        
        for j in range(0,len(im)):
            data = Image.fromarray(im[j])
            im[j]= tfms(data)
            #import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        t=np.stack(im)
        return t

    def create_train_test_split(self,split):
        # read the data
                
        #split1 = train_test_split(trainAttrX1, trainImagesX1, trainY11, test_size=0.15, random_state=42, stratify = trainY11)
        #trainAttrX, valAttrX, trainImagesX, valImagesX, trainY1, valY = split1
        mydirectory = '/content/drive/MyDrive/PhD_Laura_Santos/PW/PW_2023/neuralnetwork/DATASET' # Selected gaze
        mydirectory2 = '/content/drive/MyDrive/PhD_Laura_Santos/PW/PW_2023/neuralnetwork/DATI_PY'
        mydirectory3 = '/content/drive/MyDrive/PhD_Laura_Santos/PW/PW_2023/neuralnetwork/LABEL'
        print(mydirectory)
        gaze = pd.read_csv(mydirectory + '/Book1.csv', sep=" ", header=None)
        df = np.asarray(gaze).astype('float32')
        #elimino confidence
        #df = np.delete(df,2,1)
        #elimino headbox
        hh = np.arange(6,10)
        df = np.delete(df,hh,1)
        #elimino coordinate
        #gh = np.arange(3,6)
        #df = np.delete(df,gh,1)
        #import pdb; pdb.set_trace()
        #tolgo coordinate e hbb
        #gh = np.arange(3,10)
        #df = np.delete(df,gh,1)

        out = pd.read_csv(mydirectory3 + '/labels1.csv', sep=" ", header=None)
        out_fin = np.array(out)

        images_in = open(mydirectory2 + '/IMAGE.npy','rb')
        images = np.load(images_in, allow_pickle=True)
      
        #ind = np.arange(0, 17063,2)
        ind = np.arange(1,51505,2)


        val_data = df[ind]
        val_image = images[ind]
        val_out = out_fin[ind]

        df = np.delete(df,ind,0)
        out_fin = np.delete(out_fin,ind,0)
        images = np.delete(images,ind,0)

        # DELETE NAN
        ind_train = list()
        for i in range(0,len(df)):
            if np.isnan(df[i,0]):
                ind_train.append(i)
    
        df = np.delete(df,ind_train,0)
        out2= np.delete(out_fin,ind_train,0)
        images = np.delete(images,ind_train,0)

        ind_val = list()
        for i in range(0,len(val_data)):
            if np.isnan(val_data[i,0]):
                ind_val.append(i)

        val_data = np.delete(val_data,ind_val,0)
        val_out = np.delete(val_out,ind_val,0)
        val_image = np.delete(val_image,ind_val,0)

        #numero sample validation
        ind0 = np.where(val_out == 0)[0]
        ind1 = np.where(val_out == 1)[0]
        ind2 = np.where(val_out == 2)[0]
        ind3 = np.where(val_out == 3)[0]
        ind4 = np.where(val_out == 4)[0]
        ind5 = np.where(val_out == 5)[0]
        ind6 = np.where(val_out == 6)[0]

        m = min(len(ind0),len(ind1),len(ind2),len(ind3),len(ind4),len(ind5))

        ind0 = np.random.RandomState(seed=42).permutation(ind0)
        ind0 = ind0[:int(len(ind0)-m)]

        ind1 = np.random.RandomState(seed=42).permutation(ind1)
        ind1 = ind1[:int(len(ind1)-m)]

        ind2 = np.random.RandomState(seed=42).permutation(ind2)
        ind2 = ind2[:int(len(ind2)-m)]

        ind3 = np.random.RandomState(seed=42).permutation(ind3)
        ind3 = ind3[:int(len(ind3)-m)]

        ind4 = np.random.RandomState(seed=42).permutation(ind4)
        ind4 = ind4[:int(len(ind4)-m)]

        ind = np.concatenate((ind0, ind1,ind2,ind3,ind4,ind6), axis = 0)

        val_data = np.delete(val_data,ind,0)
        val_out = np.delete(val_out,ind,0)
        val_image = np.delete(val_image,ind,0)

        # numero di sample training e test 
        ind_0 = list()
        ind_1 = list()
        ind_2 = list()
        ind_3 = list()
        ind_4 = list()
        ind_5 = list()
        ind_6 = list()
        for i in range(0,len(out2)):
            if out2[i] == 0:
                ind_0.append(i)
            elif out2[i] == 1:
                ind_1.append(i)
            elif out2[i] == 2:
                ind_2.append(i)
            elif out2[i] == 3:
                ind_3.append(i)
            elif out2[i] == 4:
                ind_4.append(i)
            elif out2[i] == 5:
                ind_5.append(i)
            elif out2[i] == 6:
                ind_6.append(i) 

        ind_0 = np.random.RandomState(seed=42).permutation(ind_0)
        ind_0 = ind_0[:int(9186)]

        ind_1 = np.random.RandomState(seed=42).permutation(ind_1)
        ind_1 = ind_1[:int(3942)]

        ind_2 = np.random.RandomState(seed=42).permutation(ind_2)
        ind_2 = ind_2[:int(178)]

        ind_3 = np.random.RandomState(seed=42).permutation(ind_3)
        ind_3 = ind_3[:int(255)]

        ind_4 = np.random.RandomState(seed=42).permutation(ind_4)
        ind_4 = ind_4[:int(236)]

        ind = np.concatenate((ind_0, ind_1,ind_2,ind_3,ind_4,ind_6), axis = 0)

        df = np.delete(df,ind,0)
        out = np.delete(out2,ind,0)
        images = np.delete(images,ind,0)
        # output corretti 
        out_int = np.transpose(out)[0]
        out_fin = []
        for idx in range(0,len(out_int)):
            action = int(out_int[idx])
            
            one_hot_vector = np.zeros(6)
            one_hot_vector[action] = 1
            out_fin.extend(one_hot_vector)
        output_att = np.array(out_fin)
        output_finale = np.reshape(output_att,(-1,6))
        # output_finale = np.transpose(out1)

        val_int = np.transpose(val_out)[0]
        val_out_fin = []
        for idx2 in range(0,len(val_int)):
            action2 = int(val_int[idx2])
            one_hot_vector2 = np.zeros(6)
            one_hot_vector2[action2] = 1
            val_out_fin.extend(one_hot_vector2)
        val_output_att = np.array(val_out_fin)
        val_output_finale = np.reshape(val_output_att,(-1,6))


        cs = ColumnTransformer([("nome", MinMaxScaler(),[0,1,2]),("nome2", MinMaxScaler(),[3,4,5])])


        trainContinuous = cs.fit_transform(df)

        X = trainContinuous
        df[:,0:6]=X


        #validation set
        # cs2 = ColumnTransformer([("nome", MinMaxScaler(),[0,1,2]),("nome2", MinMaxScaler(),[3,4,5])])
        trainContinuous2 = cs.fit_transform(val_data)

        X2 = trainContinuous2
        val_data[:,0:6]=X2


        #scal e pad im
        image_fin = self.norm_imm(images)
        val_image_fin = self.norm_imm(val_image)

        split = train_test_split(df, image_fin, output_finale, test_size=0.1, random_state=42, stratify = out)
        trainAttrX, testAttrX, trainImagesX, testImagesX, trainY1, testY1 = split
        data = trainAttrX, testAttrX, trainImagesX, testImagesX, trainY1, testY1, val_data, val_image_fin, val_output_finale

        with open('~/Desktop/Patients/Pickles/no_hbb_me.pkl', 'wb') as f:
            pickle.dump(data, f)

    def read_data(self, label):

        # import pdb; pdb.set_trace()
        if label == 'train':
            trainAttrX, trainImagesX, trainY1 = pickle.load(open('/content/drive/MyDrive/PhD_Laura_Santos/PW/PW_2023/pickles/no_hbb_me_train.pkl', 'rb'), encoding='latin1')

            self.gaze = trainAttrX
            self.immagine = trainImagesX
            self.output = trainY1
            # import pdb; pdb.set_trace()
        if label == 'test':
            testAttrX, testImagesX, testY1 = pickle.load(open('/content/drive/MyDrive/PhD_Laura_Santos/PW/PW_2023/pickles/no_hbb_me_test.pkl', 'rb'), encoding='latin1')

            self.gaze = testAttrX
            self.immagine = testImagesX
            self.output = testY1
        if label == 'val':
            val_data, val_image_fin, val_output_finale = pickle.load(open('/content/drive/MyDrive/PhD_Laura_Santos/PW/PW_2023/pickles/no_hbb_me_val.pkl', 'rb'), encoding='latin1')

            self.gaze = val_data
            self.immagine = val_image_fin
            self.output = val_output_finale
            # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.gaze)

    def __getitem__(self, idx): 
        gaze = self.gaze[idx]
        output = self.output[idx]
        immagine = self.immagine[idx]
        return torch.tensor(gaze).float(), torch.tensor(immagine), torch.tensor(output).float()