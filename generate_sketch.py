import sys,os
from PIL import Image    
from dcgan import DCGAN
import numpy as np

import tensorflow as tf
print(tf.__version__)
batch_size = 32
n_noise = 100
dcgan = DCGAN(batch_size, n_noise)
#
dataset = []
dir_name = 'sketch'
for name in os.listdir(dir_name):
    if not name.startswith('.'):
        path = dir_name + '/'  + name
        img = Image.open(path)
        img = img.resize((64,64))
        img = np.array(img,dtype=np.float32)
#        img = (img - 127.5) / 127.5
        
        if img.shape == (64, 64,3):
            dataset.append(img)
dataset = np.array(dataset)
dcgan.train(dataset,1000)

