import tensorflow as tf
import numpy as np
from PIL import Image
import sys,os
print(tf.__version__)

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
                  activation='relu',
                  kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
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
                   kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
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
                   kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
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
                   kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
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
                   activation='tanh',
                   kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
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
                  padding='same',
                  kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU(alpha=0.2)
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )



        discriminator.add(
               tf.keras.layers.Conv2D(
                  filters=self.n_w3,
                  kernel_size=(5,5),
                  strides=(2,2),
                  padding='same',
                  kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU(alpha=0.2)
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )


        discriminator.add(
               tf.keras.layers.Conv2D(
                  filters=self.n_w2,
                  kernel_size=(5,5),
                  strides=(2,2),
                  padding='same',
                  kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU(alpha=0.2)
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )


        discriminator.add(
               tf.keras.layers.Conv2D(
                  filters=self.n_w1,
                  kernel_size=(5,5),
                  strides=(2,2),
                  padding='same',
                  kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02)
                   ) 
                )
        discriminator.add(
                tf.keras.layers.LeakyReLU(alpha=0.2)
                )
        discriminator.add(
                tf.keras.layers.BatchNormalization()
                )
        discriminator.add(
               tf.keras.layers.Flatten() 
                )
        discriminator.add(
               tf.keras.layers.Dense(1,kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02) ) 
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

    def train(self,dataset,epochs):
        from tqdm import tqdm
        dis_opt = tf.keras.optimizers.Adam(1e-4)
        gen_opt = tf.keras.optimizers.Adam(1e-4)
        for epoch in tqdm(range(epochs)):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                batch = np.random.choice(dataset.shape[0],self.batch_size)
                batch_images = dataset[batch]
                noise = np.random.uniform(-1,1,[self.batch_size,100])
                generated_image = self.generator(noise)
                fake_output = self.discriminator(generated_image)
                real_output = self.discriminator(batch_images)
                dis_loss = self.discriminator_loss(real_output, fake_output)
                gen_loss = self.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(dis_loss, self.discriminator.trainable_variables)
            gen_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            dis_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            
            if epoch % 10 == 0:
                noise_sample = np.random.uniform(-1,1,[1,100])
                generated_image = self.generator(noise_sample)
                generated_image = generated_image.numpy()[0]
                generated_image = Image.fromarray(((127.5+generated_image) * 127.5).astype(np.uint8))
#                generated_image = Image.fromarray(generated_image.astype(np.uint8))

                try :
                    os.mkdir('generated_image')
                    generated_image.save('generated_image/out_at_{}.jpg'.format(str(epoch)))
                except :
                    generated_image.save('generated_image/out_at_{}.jpg'.format(str(epoch)))


if __name__ == '__main__':
    batch_size = 128
    n_noise = 100
    dcgan = DCGAN(batch_size, n_noise)
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
            img = (img - 127.5) / 127.5
            dataset.append(img)
    dataset = np.array(dataset)
    
    dcgan.train(dataset,1000)



