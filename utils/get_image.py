from PIL import Image
import os
import imageio
import numpy as np

parent_dir = '/Users/lostinstaches/Desktop/Patients' # LOOK HERE
for patient_folder in os.listdir(parent_dir):
    # print(patient_folder)
    patient_path = os.path.join(parent_dir, patient_folder)

    # Skip if it's not a directory
    if not os.path.isdir(patient_path):
        continue # Extract all the files in a directory

    gaze_files = [f for f in os.listdir(patient_path) if 'selected_gaze' in f]
    if not gaze_files:
        print(f"No gaze file found in {patient_folder}")
        continue

    gaze_file_path = os.path.join(patient_path, gaze_files[0])

    # gaze = open(patient_path + '/selected_gaze_ch.txt').read()
    gaze = open(gaze_file_path).read()
    listafiles = os.listdir(patient_path)  # Extract all the files in a directory
    vecvid = [s for s in listafiles if 'video1' in s]

    video = imageio.get_reader(patient_path + '/' + vecvid[0])

    f = open(patient_path + '/IMAGE.npy', 'wb')
    # gaze = gaze.replace("/n", " /n ")
    y = gaze.splitlines()

    print(y)
    print("Test")
    print(y[0])
    v = []
    i = 0
    imglist = list()

    for n, l in enumerate(y):
        # Reads line
        x = l.split(" ")

        # If even number, reads next frame of video
        if (n % 2) == 0:
            image = video.get_next_data()
            im = Image.fromarray(image.astype('uint8'))
        if not "NaN" in x[0]:
            box = [x[6], x[7], x[8], x[9]]
            print(x[6], x[7], x[8], x[9])

            new_im = im.crop(
                (int(x[6]) * 1920 / 960, int(x[7]) * 1080 / 720, int(x[8]) * 1920 / 960, int(x[9]) * 1080 / 720))
            # Creates array

            new_im = new_im.resize((100, 120))  # Resize to a fixed size
            imgarray = np.asarray(new_im)
            imglist.append(imgarray)
            # print(new_im.size)

            # imgarray = np.asarray(new_im)
            # imglist.append(imgarray)


    np.save(f, imglist, allow_pickle=True)
    f.close()

    f = open(patient_path + '/IMAGE.npy', 'rb')
    imgarrayfile = np.load(f, allow_pickle=True)
    print(imgarrayfile.size)