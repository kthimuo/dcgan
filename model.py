import numpy as np
import tensorflow as tf

def batch_norm(X,scale,offset,axes,is_train,deveice_name='/gpu:0'):
    if is_train is False:
        return X

    epsilon = 1e-5
    with tf.device(device_name):
        mean, variance = tf.moments(X, axes)
        bn = tf.nn.batch_normalization(X, maen, variance, offset, scale,epsilon)
    return bn

class Generator():
    def __init__(self,device_name='/gpu:0'):
        with tf.device(device_name):
            self.gen_w0 = tf.Variable(
                    tf.random.normal(shape=[100,4*4*1024],stddev=0.02,dtype=tf.float32,
                        name='gen_w0'))

            self.gen_b0 = tf.Variable(
                    tf.random.normal(shape=[4*4*1024],stddev=0.02,dtype=tf.float32,
                        name='gen_b0'))

            self.gen_w1 = tf.Variable(
                   tf.random.normal(shape=[5,5,512,1024],
                       stddev=0.02, dtype=tf.float32),
                   name='gen_w1'
                    )

            self.gen_b1 = tf.Variable(
                   tf.random.normal(shape=[128],
                       stddev=0.02, dtype=tf.float32),
                   name='gen_w1'
                    )

            self.gen_w2 = tf.Variable(
                tf.random.normal(
                    shape=[4, 4, 64, 128], stddev=0.02, dtype=tf.float32),
                name="gen_w2")

            self.gen_b2 = tf.Variable(
                tf.random.normal(
                    shape=[64], stddev=0.02, dtype=tf.float32),
                name="gen_b2")

            self.gen_w3 = tf.Variable(
                tf.random.normal(
                    shape=[4, 4, 1, 64], stddev=0.02, dtype=tf.float32),
                name="gen_w3")

            self.gen_b3 = tf.Variable(
                tf.random.normal(
                    shape=[1], stddev=0.02, dtype=tf.float32),
                name="gen_b3")

            self.gen_scale_w1 = tf.Variable(
                tf.ones([128]), name="gen_scale_w1")
            self.gen_offset_w1 = tf.Variable(
                tf.zeros([128]), name="gen_offset_w1")

            self.gen_scale_w2 = tf.Variable(
                tf.ones([64]), name="gen_scale_w2")
            self.gen_offset_w2 = tf.Variable(
                tf.zeros([64]), name="gen_offset_w2")

#            self.keep_prob = tf.placeholder(tf.float32)
#            self.batch_size = tf.placeholder(tf.int32)
            
            self.keep_prob = tf.Variable(tf.constant(1.0),name='keep_prob')
#            self.batch_size = tf.Variable(tf.constant(100name='batch_size')

            self.batch_size = 32
    def run(self, z, is_train, device_name='/gpu:0'):

        with tf.device(device_name):
            h0  = tf.nn.relu(tf.matmul(z,self.gen_w0) + self.gen_b0)
            print('======')
            print(h0.shape)

            h0  = tf.reshape(h0,[32,4, 4,1024])
#            gen_conv1 = tf.nn.conv2d_transpose(
#                    h0,
#                    filters=self.gen_w1,
#                    output_shape=[32,8,8,512],
#                    strides=[1, 2, 2, 1],
#                    padding='SAME')
#            gen_conv1 = tf.keras.layers.BatchNormalization(gen_conv1, scale=0.9,epsilon=1e-5)
#            gen_conv1 = tf.nn.relu(gen_conv1)
#
#            gen_conv2 = tf.nn.conv2d_transpose(
#                   gen_conv1,
#                   filters=self.gen_w2,
#                   output_shape=[32,16,16,256],
#                    strides=[1, 2, 2, 1],
#                    padding='SAME')
#            gen_conv2 = tf.nn.batch_normalization(gen_conv2, scale=0.9,variance_epsilon=1e-5)
#            gen_conv2 = tf.nn.relu(gen_conv2)
#
#            gen_conv3 = tf.nn.conv2d_transpose(
#                   gen_conv2,
#                   filters=self.gen_w3,
#                   output_shape=[32,32,32,128],
#                    strides=[1, 2, 2, 1],
#                    padding='SAME')
#            gen_conv3 = tf.nn.batch_normalization(gen_conv3, scale=0.9,variance_epsilon=1e-5)
#            gen_conv3 = tf.nn.relu(gen_conv3)
#
#            gen_conv4 = tf.nn.conv2d_transpose(
#                   gen_conv3,
#                   filters=self.gen_w4,
#                   output_shape=[32,64,64,3],
#                    strides=[1, 2, 2, 1],
#                    padding='SAME')
#            print('======')
#            print(gen_conv4.shape)


                    
                    

            







            





if __name__ == '__main__':
#    device_name =tf.test.gpu_device_name()
    g = Generator()
#    print(g.gen_w0.shape)

#    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
#    print(x_train.shape)
    noize_z = np.random.uniform(-1,1,size=[32,100]).astype(np.float32)
    
    z_const = tf.constant(noize_z,dtype=tf.float32)
    g.run(z_const,is_train=False)
    
#    gen_w0 = tf.Variable(
#            tf.random.normal(shape=[100,4*4*1024],stddev=0.02,dtype=tf.float32,
#                name='gen_w0'))
#    tf.matmul(z_const,gen_w0)


  

