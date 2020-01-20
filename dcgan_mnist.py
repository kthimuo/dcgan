import tensorflow as tf
import numpy as np

class DCGAN:
    def __init__(self, batch_size, n_noise, image_size, image_channel):
        '''
        n_noise : uniform_random で生成されるノイズの個数(ノイズ1セットに対して1枚出力)
        image_size : Generatorの出力する画像のサイズ
        '''
        self.batch_size = batch_size
        self.n_noise = n_noise
        self.image_size = image_size
        self.image_channel = image_channel

        self.n_input = image_size*image_size*image_channel
        self.n_w1 = 1024
        self.n_w2 = 512
        self.n_w3 = 256
        self.n_w4 = 128
        self.n_w5 = 3
        self.n_hidden = 4*4*self.n_w1

        self.epsilon = 1e-5
        self.momentum = 0.9
        self.alpha = 0.2

    def input(self):
        x = tf.constant(tf.float32,[None, self.image_size, self.image_size,self.image_channel])
        z = tf.constant(tf.float32, [None,self.n_noise])
        lr = tf.constant(tf.float32,[])
        return x,z,lr

    def generator(self,noise_z,reuse=False):
        with tf.name_scope('generator'):
            G_FW1 = tf.Variable(
                    tf.random.normal(shape=(self.n_noise,self.n_hidden),
                    mean=0.0,
                    stddev=0.02,
                    dtype=tf.dtypes.float32),
                    name='G_FW1')
            G_Fb1 = tf.constant(
                    0,
                    shape=[self.n_hidden],
                    name='G_Fb1')
            G_W1 = tf.Variable(
                    tf.random.normal(shape=(5,5,self.n_w2,self.n_w1),
                    mean=0.0,
                    stddev=0.02,
                    dtype=tf.dtypes.float32),
                    name='G_W1'
                    )
            G_W2= tf.Variable(
                    tf.random.normal(shape=(5,5,self.n_w3,self.n_w2),
                    mean=0.0,
                    stddev=0.02,
                    dtype=tf.dtypes.float32),
                    name='G_W2'
                    )
            G_W3= tf.Variable(
                    tf.random.normal(shape=(5,5,self.n_w4,self.n_w3),
                    mean=0.0,
                    stddev=0.02,
                    dtype=tf.dtypes.float32),
                    name='G_W3'
                    )
            G_W4= tf.Variable(
                    tf.random.normal(shape=(5,5,self.n_w5,self.n_w4),
                    mean=0.0,
                    stddev=0.02,
                    dtype=tf.dtypes.float32),
                    name='G_W4'
                    )
        generator = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_hidden,input_shape=[self.batch_size,self.n_noise]),
            tf.keras.layers.Reshape((4, 4, self.n_w1)),
            
            tf.keras.layers.Conv2DTranspose(self.n_w2,(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(self.n_w3,(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(self.n_w4,(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(self.n_w5,(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('tanh')
            ])
        out = generator(noise_z)
        print('=========')
        print(out.shape)
        return out

    def discriminator(self,inp,reuse=False):
        with tf.name_scope('discriminator'):
            D_W1 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.image_channel,self.n_w5),
                        stddev=0.02),
                    name='D_W1'
                    )
            D_W2 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.n_w5.self.ne_w4),
                        stddev=0.02),
                    name='D_W2'
                    )
            D_W3 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.n_w4.self.ne_w3),
                        stddev=0.02),
                    name='D_W2'
                    )
            D_W2 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.n_w3.self.ne_w2),
                        stddev=0.02),
                    name='D_W2'
                    )




        discriminator = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.n_w4,kernel_size=(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(epsilon=self.epsilon,momentum=self.momentum),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Conv2D(self.n_w3,(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(epsilon=self.epsilon,momentum=self.momentum),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Conv2D(self.n_w2,(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(epsilon=self.epsilon,momentum=self.momentum),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Conv2D(self.n_w1,(5,5),strides=(2,2),padding='same'),
            tf.keras.layers.BatchNormalization(epsilon=self.epsilon,momentum=self.momentum),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Reshape((4*4*self.n_w1,),
                input_shape=(self.batch_size,4,4,1024)),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
            ])
        out = discriminator(inp)
        return out

            
            
       
if __name__ == '__main__':
    batch_size = 32
    n_noise = 100
    image_size = 64
    image_chanel = 3

    noise_z = np.random.uniform(-1,1,size=[32,100]).astype(np.float32)
#
    dcgan = DCGAN(batch_size, n_noise, image_size, image_chanel)
    g_out = dcgan.generator(noise_z)
    d_out = dcgan.discriminator(g_out)

