import os, os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np



imgs = []
path = "images/"
valid_images = [".jpg"] # ,".gif"
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() == '.jpg':
        
        imgs.append(cv2.imread((os.path.join(path,f))))
        
        
plt.imshow(imgs[5])        

div = np.array(imgs)
div
