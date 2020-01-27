import tensorflow as tf
import numpy as np

class DCGAN:
    def __init__(self, batch_size, n_noise):
        self.batch_size = batch_size
        self.n_noise = n_noise

        self.n_w1 = 1024 
        self.n_w2 = 512
        self.n_w3 = 256
        self.n_w4 = 128
        self.n_w5 = 3

        self.generator = self.make_generator()

    def make_generator(self):
        generator = tf.keras.Sequential()
        generator.add(
               tf.keras.layers.Dense(units=4*4*self.n_w1,
                  input_shape=(self.n_noise,),  
                  activation='relu'
                  )
                )
        generator.add(
               tf.keras.layers.Reshape(
                   (4,4,self.n_w1)
                   )
                )
        generator.add(
                tf.keras.layers.ReLU()
                )
        generator.add(
                tf.keras.layers.BatchNormalization()
                )

        generator.add(
               tf.keras.layers.Conv2DTranspose(filters=self.n_w2,
                   kernel_size=5,
                   strides=1,
                   )
            )
        generator.add(
                tf.keras.layers.ReLU()
                )
        generator.add(
                tf.keras.layers.BatchNormalization()
                )
        

        generator.add(
               tf.keras.layers.Conv2DTranspose(filters=self.n_w3,
                   kernel_size=(5,5),
                   strides=(2,2),
                   padding='same',
                   ),
               )
        generator.add(
                tf.keras.layers.ReLU()
                )
        generator.add(
                tf.keras.layers.BatchNormalization()
                )


        generator.add(
               tf.keras.layers.Conv2DTranspose(filters=self.n_w4,
                   kernel_size=(5,5),
                   strides=(2,2),
                   padding='same',
                   )
               )
        generator.add(
                tf.keras.layers.ReLU()
                )
        generator.add(
                tf.keras.layers.BatchNormalization()
                )
        generator.add(
               tf.keras.layers.Conv2DTranspose(filters=self.n_w5,
                   kernel_size=(5,5),
                   strides=(2,2),
                   padding='same',
                   activation='tanh'
                   )
               )
               
        return generator


if __name__ == '__main__':
    batch_size = 128
    n_noise = 100


    noise = np.random.uniform(-1,1,(batch_size, 100))
    print(noise.shape)
    dcgan = DCGAN(batch_size, n_noise)
    out = dcgan.generator(noise)
    print(out.shape)


        

