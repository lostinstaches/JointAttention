import numpy as np
import os
from sklearn.model_selection import train_test_split

import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import torchvision.transforms as transforms
from PIL import Image
image_size = 224
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


def norm_imm(self, im):
    for j in range(0, len(im)):
        data = Image.fromarray(im[j])
        im[j] = tfms(data)
        # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    t = np.stack(im)
    return t

patients_dir = '/Users/lostinstaches/Desktop/Patients'  # Selected gaze

gaze = pd.read_csv(patients_dir + '/Book1.csv', sep=" ", header=None)
df = np.asarray(gaze).astype('float32')

print("Shape of df:", df.shape)
print("Size of df:", df.size)

hh = np.arange(6, 10)
print(hh)
df = np.delete(df, hh, 1)
print("Shape of df:", df.shape)
print("Size of df:", df.size)

out = pd.read_csv(patients_dir + '/labels1.csv', sep=" ", header=None)
out_fin = np.array(out)

print("Shape of out_fin:", out_fin.shape)
print("Size of out_fin:", out_fin.size)


# print(os.listdir(patients_dir + '/'))

print(os.listdir('/Users/lostinstaches/Desktop/Patients'))
images_in = open(patients_dir + '/image.npy', 'rb')
images = np.load(images_in, allow_pickle=True)

# ind = np.arange(0, 17063,2)


# ind = np.arange(0, 17970, 2)
ind = np.arange(0, 595, 2)
print(df.size)
val_data = df[ind]
val_image = images[ind]
val_out = out_fin[ind]

df = np.delete(df, ind, 0)
out_fin = np.delete(out_fin, ind, 0)
images = np.delete(images, ind, 0)

# DELETE NAN
ind_train = list()
for i in range(0, len(df)):
    if np.isnan(df[i, 0]):
        ind_train.append(i)

df = np.delete(df, ind_train, 0)
out2 = np.delete(out_fin, ind_train, 0)
images = np.delete(images, ind_train, 0)

ind_val = list()
for i in range(0, len(val_data)):
    if np.isnan(val_data[i, 0]):
        ind_val.append(i)

val_data = np.delete(val_data, ind_val, 0)
val_out = np.delete(val_out, ind_val, 0)
val_image = np.delete(val_image, ind_val, 0)

# numero sample validation
ind0 = np.where(val_out == 0)[0]
ind1 = np.where(val_out == 1)[0]
ind2 = np.where(val_out == 2)[0]
ind3 = np.where(val_out == 3)[0]
ind4 = np.where(val_out == 4)[0]
ind5 = np.where(val_out == 5)[0]
ind6 = np.where(val_out == 6)[0]
# ADD HERE

m = min(len(ind0), len(ind1), len(ind2), len(ind3), len(ind4), len(ind5))

ind0 = np.random.RandomState(seed=42).permutation(ind0)
ind0 = ind0[:int(len(ind0) - m)]

ind1 = np.random.RandomState(seed=42).permutation(ind1)
ind1 = ind1[:int(len(ind1) - m)]

ind2 = np.random.RandomState(seed=42).permutation(ind2)
ind2 = ind2[:int(len(ind2) - m)]

ind3 = np.random.RandomState(seed=42).permutation(ind3)
ind3 = ind3[:int(len(ind3) - m)]

ind4 = np.random.RandomState(seed=42).permutation(ind4)
ind4 = ind4[:int(len(ind4) - m)]

ind = np.concatenate((ind0, ind1, ind2, ind3, ind4, ind6), axis=0)

val_data = np.delete(val_data, ind, 0)
val_out = np.delete(val_out, ind, 0)
val_image = np.delete(val_image, ind, 0)

# numero di sample training e test
ind_0 = list()
ind_1 = list()
ind_2 = list()
ind_3 = list()
ind_4 = list()
ind_5 = list()
ind_6 = list()
for i in range(0, len(out2)):
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

ind = np.concatenate((ind_0, ind_1, ind_2, ind_3, ind_4, ind_6), axis=0)

df = np.delete(df, ind, 0)
out = np.delete(out2, ind, 0)
images = np.delete(images, ind, 0)
# output corretti
out_int = np.transpose(out)[0]
out_fin = []
for idx in range(0, len(out_int)):
    action = int(out_int[idx])
    # TODO: CHange to the number of targets
    one_hot_vector = np.zeros(7)
    one_hot_vector[action] = 1
    out_fin.extend(one_hot_vector)
output_att = np.array(out_fin)
output_finale = np.reshape(output_att, (-1, 7))
# output_finale = np.transpose(out1)

val_int = np.transpose(val_out)[0]
val_out_fin = []
for idx2 in range(0, len(val_int)):
    action2 = int(val_int[idx2])
    one_hot_vector2 = np.zeros(7)
    one_hot_vector2[action2] = 1
    val_out_fin.extend(one_hot_vector2)
val_output_att = np.array(val_out_fin)
val_output_finale = np.reshape(val_output_att, (-1, 7))

cs = ColumnTransformer([("nome", MinMaxScaler(), [0, 1, 2]), ("nome2", MinMaxScaler(), [3, 4, 5])])

trainContinuous = cs.fit_transform(df)

X = trainContinuous
df[:, 0:7] = X

# validation set
# cs2 = ColumnTransformer([("nome", MinMaxScaler(),[0,1,2]),("nome2", MinMaxScaler(),[3,4,5])])
trainContinuous2 = cs.fit_transform(val_data)

X2 = trainContinuous2
val_data[:, 0:7] = X2

# scal e pad im
image_fin = norm_imm(images)
val_image_fin = norm_imm(val_image)

split = train_test_split(df, image_fin, output_finale, test_size=0.1, random_state=42, stratify=out)
trainAttrX, testAttrX, trainImagesX, testImagesX, trainY1, testY1 = split
data = trainAttrX, testAttrX, trainImagesX, testImagesX, trainY1, testY1, val_data, val_image_fin, val_output_finale

with open('~/Desktop/Patients/Pickles/no_hbb_me.pkl', 'wb') as f:
    pickle.dump(data, f)