from cmath import nan
from turtle import end_fill
from PIL import Image
import os
import cv2
import imageio
import numpy as np


mydirectory = '/Users/lostinstaches/Desktop/Gabriele_Laura/p1'
# mydirectory2 = 'C:\\Users\\laura\\Documents\\Dados\\Arianna_Samuele\\p2\\IMAGES'
# mydirectory3 = 'C:\\Users\\laura\\Documents\\Dados\\Arianna_Samuele\\p2\\GAZE'

listafiles = os.listdir(mydirectory) # Extract all the files in a directory

gaze = open(mydirectory + '/selected_gaze_ch.txt').read()
listafiles = os.listdir(mydirectory) # Extract all the files in a directory
vecvid     = [s for s in listafiles if 'video1' in s]

video = imageio.get_reader(mydirectory + '/'+vecvid[0])

f = open(mydirectory + '/IMAGE.npy', 'wb')

y = gaze.splitlines()
v = []
i = 0
imglist = list()
for n, l in enumerate(y):
    # Doesn't read last 2 lines of the file
    # if n == len(y)-2:
    #     break
    # Reads line
    x = l.split(" ")
    # If even number, reads next frame of video
    if (n%2) == 0:
        image = video.get_next_data()
        im = Image.fromarray(image.astype('uint8'))
    if not "NaN" in x[0]:
        box = [x[6],x[7],x[8],x[9]]
        new_im = im.crop((int(x[6])*1920/960,int(x[7])*1080/720,int(x[8])*1920/960,int(x[9])*1080/720))
        # Creates array
        imgarray = np.asarray(new_im)
        imglist.append(imgarray)
        #if n == 1300:
        #    new_im.save("gazenew" + str(n) + ".png")
    # else:
    #     imglist.append(np.nan)

np.save(f, imglist, allow_pickle=True)
f.close()

f = open(mydirectory + '/IMAGE.npy', 'rb')
imgarrayfile = np.load(f, allow_pickle=True)
print(imgarrayfile.size)
#
#im = Image.fromarray(imgarrayfile[1300].astype('uint8'))
#im.save("gazeload.png")