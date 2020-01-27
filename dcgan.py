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
        self.discriminator = self.make_discriminator()

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

    def make_discriminator(self):
        discriminator = tf.keras.Sequential()
        discriminator.add(
               tf.keras.layers.Conv2D(
                  filters=self.n_w4,
                  kernel_size=(5,5),
                  strides=(2,2),
                  padding='same'
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU()
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )



        discriminator.add(
               tf.keras.layers.Conv2D(
                  filters=self.n_w3,
                  kernel_size=(5,5),
                  strides=(2,2),
                  padding='same'
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU()
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )


        discriminator.add(
               tf.keras.layers.Conv2D(
                  filters=self.n_w2,
                  kernel_size=(5,5),
                  strides=(2,2),
                  padding='same'
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU()
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )


        discriminator.add(
               tf.keras.layers.Conv2D(
                  filters=self.n_w1,
                  kernel_size=(5,5),
                  strides=(2,2),
                  padding='same'
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU()
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )
        discriminator.add(
               tf.keras.layers.Flatten() 
                )
        discriminator.add(
               tf.keras.layers.Dense(1) 
                )
        return discriminator

    def discriminator_loss(self,real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def train(self,images,epochs):
        dis_opt = tf.keras.optimizers.Adam(1e-4)
        gen_opt = tf.keras.optimizers.Adam(1e-4)
        for epoch in range(epochs):
            batch = np.random.choice(dataset.shape[0],self.batch_size)
            batch_images = images[batch]
            noise = np.random.uniform(-1,1,[self.batch_size,100])
            generated_image = self.generator(noise)
            
            fake_output = self.discriminator(generated_image)
            real_output = self.discriminator(batch_images)
            
            dis_loss = self.discriminator_loss(real_output, fake_output)
            gen_loss = self.generator_loss(fake_output)

            a = self.discriminator.trainable_variables
            print(a)
            break
#            dis_opt.minimize(dis_loss,])
#            print('ok')
#            break


            

if __name__ == '__main__':
    batch_size = 128
    n_noise = 100


#    noise = np.random.uniform(-1,1,(batch_size, 100))
#    print(noise.shape)
    dcgan = DCGAN(batch_size, n_noise)
#    out = dcgan.generator(noise)
#    inver = dcgan.discriminator(out)
#    print(inver.shape)
#
    dataset = []
    import sys,os
    from PIL import Image    
    dir_name = 'apples'
    for name in os.listdir(dir_name):
        if not name.startswith('.'):
            path = dir_name + '/' + name
            img = Image.open(path)
            img = img.resize((64,64))
            img = np.array(img,dtype=np.float32)
            dataset.append(img)
    dataset = np.array(dataset)
    
    dcgan.train(dataset,100)



