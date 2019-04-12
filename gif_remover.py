import os
import cv2

# folder = './tensor_classifier/training_dataset/'
folder = "/Users/Bipen/Documents/University/*CSC420/Project/OTHER/Scarlett_Johansson"

for subdir, dirs, files in os.walk(folder):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        if file.endswith('.jpg'):
            try:
                img = cv2.imread(filepath)
                print(img.shape)
            except AttributeError:
                os.remove(filepath)
