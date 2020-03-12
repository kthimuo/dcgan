import sys,os
from PIL import Image    
from dcgan import DCGAN
import numpy as np
batch_size = 512
n_noise = 100
dcgan = DCGAN(batch_size, n_noise)
#
dataset = []
dir_name = 'fruits'
for fruit_name in os.listdir(dir_name):
    if not fruit_name.startswith('.'):
        for name in os.listdir(dir_name + '/' + fruit_name):
            if not name.startswith('.'):
                path = dir_name + '/' + fruit_name + '/' + name
                img = Image.open(path)
                img = img.resize((64,64))
                img = np.array(img,dtype=np.float32)
                img = (img - 127.5) / 127.5
                dataset.append(img)
dataset = np.array(dataset)
print(dataset.shape)
dcgan.train(dataset,10000)
