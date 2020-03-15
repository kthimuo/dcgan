import sys,os
from PIL import Image    
from dcgan import DCGAN
import numpy as np
batch_size = 128
n_noise = 100
dcgan = DCGAN(batch_size, n_noise)

dataset = []
dir_name = 'apples'

for name in os.listdir(dir_name):
    if not name.startswith('.'):
        path = dir_name + '/'  + name
        img = Image.open(path)
        img = img.resize((64,64))
        img = np.array(img,dtype=np.float32)
        img = (img - 127.5) / 127.5
        dataset.append(img)
dataset = np.array(dataset)
dcgan.train(dataset,100)

