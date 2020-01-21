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
        self.n_w5 = 64
    
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
        with tf.name_scope('generator') as scope:
            G_FW1 = tf.Variable(
                    tf.random.normal(shape=(self.n_noise,self.n_hidden),
                    mean=0.0,
                    stddev=0.02,
                    dtype=tf.dtypes.float32),
                    name='G_FW1')
            G_Fb1 = tf.constant(
                    0.0,
                    shape=[self.n_hidden],
                    dtype=tf.dtypes.float32,
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
                    tf.random.normal(shape=(5,5,self.image_channel,self.n_w4),
                    mean=0.0,
                    stddev=0.02,
                    dtype=tf.dtypes.float32),
                    name='G_W4'
                    )

        hidden = tf.nn.relu(tf.matmul(noise_z,G_FW1) + G_Fb1)
        hidden = tf.reshape(hidden,shape=(self.batch_size,4,4,self.n_w1))
        dconv1 = tf.nn.conv2d_transpose(hidden,G_W1,
                output_shape=(self.batch_size,8,8,self.n_w2),
                strides=[1,2,2,1])
        dconv1 = tf.nn.batch_normalization(dconv1,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        dconv1 = tf.nn.relu(dconv1)

        dconv2 = tf.nn.conv2d_transpose(dconv1,G_W2,
                output_shape=(self.batch_size,16,16,self.n_w3),
                strides=[1,2,2,1])
        dconv2 = tf.nn.batch_normalization(dconv2,mean=0,variance=0.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        dconv2 = tf.nn.batch_normalization(dconv2,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        dconv2 = tf.nn.relu(dconv2)

        dconv3 = tf.nn.conv2d_transpose(dconv2,G_W3,
                output_shape=(self.batch_size,32,32,self.n_w4),
                strides=[1,2,2,1])
        dconv3 = tf.nn.batch_normalization(dconv3,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        dconv3 = tf.nn.relu(dconv3)

        dconv4 = tf.nn.conv2d_transpose(dconv3,G_W4,
                output_shape=(self.batch_size,64,64,self.image_channel),
                strides=[1,2,2,1])
        dconv4 = tf.nn.batch_normalization(dconv4,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        out = tf.nn.tanh(dconv4)
        return out
    def discriminator(self,inp,reuse=False):

        with tf.name_scope('discriminator'):
            D_W1 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.image_channel,self.n_w4),
                        stddev=0.02),
                    name='D_W1'
                    )
            D_W2 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.n_w4,self.n_w3),
                        stddev=0.02),
                    name='D_W2'
                    )
            D_W3 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.n_w3,self.n_w2),
                        stddev=0.02),
                    name='D_W3'
                    )
            D_W4 = tf.Variable(
                    tf.random.truncated_normal(shape=(5,5,self.n_w2,self.n_w1),
                        stddev=0.02),
                    name='D_W4'
                    )
            D_FW1 = tf.Variable(
                    tf.random.truncated_normal(shape=(4*4*self.n_w1,1))
                    )

        conv1 = tf.nn.conv2d(inp,D_W1,strides=[1,2,2,1],padding='SAME')
        conv1 = tf.nn.batch_normalization(conv1,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        conv1 = tf.nn.leaky_relu(conv1)
       
        conv2 = tf.nn.conv2d(conv1,D_W2,strides=[1,2,2,1],padding='SAME')
        conv2 = tf.nn.batch_normalization(conv2,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        conv2 = tf.nn.leaky_relu(conv2)


        conv3 = tf.nn.conv2d(conv2,D_W3,strides=[1,2,2,1],padding='SAME')
        conv3 = tf.nn.batch_normalization(conv3,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        conv3 = tf.nn.leaky_relu(conv3)

        conv4 = tf.nn.conv2d(conv3,D_W4,strides=[1,2,2,1],padding='SAME')
        conv4 = tf.nn.batch_normalization(conv4,mean=0,variance=1.0,offset=0.0,scale=0.9,variance_epsilon=1e-5)
        conv4 = tf.nn.leaky_relu(conv4)

        hidden = tf.reshape(conv4,[self.batch_size,4*4*self.n_w1])
        output = tf.matmul(hidden,D_FW1)
        output = tf.sigmoid(output)
        a = tf.Graph().get_all_collection_keys()
        
        print('============')
        print(tf.Variable.variables())
        print(a)
        print('============')
        return output

    def loss(self,X,Z):
        g_out = self.generator(Z)
        d_fake = self.discriminator(g_out)
        d_real = self.discriminator(X)

        d_loss = tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake))
        g_loss = tf.reduce_mean(tf.log(d_fake))
        return d_loss, g_loss

#    def optimizer(self,d_loss,g_loss,learning_rate):
#        d_var_list = tf.Graph.collections()
    def optimizer(self):
#        a = tf.Graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='generator')
        a = tf.Graph().get_all_collection_keys()
        print(a)

            
            
       
if __name__ == '__main__':
    batch_size = 128
    n_noise = 100
    image_size = 64
    image_chanel = 3

    noise_z = np.random.uniform(-1,1,size=[batch_size,100]).astype(np.float32)
#
    dcgan = DCGAN(batch_size, n_noise, image_size, image_chanel)

    
    g_out = dcgan.generator(noise_z)
    d_out = dcgan.discriminator(g_out)
    dcgan.optimizer()

