import numpy as np
import matplotlib.pyplot as plt
import cv2
import string
import random

import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    losses,
    utils,
    metrics,
    optimizers,
)
from utils import display, sample_batch
################################################


class DCGANGenerator:

    def __init__(self): 
        self.IMAGE_SIZE = 64
        self.CHANNELS = 3
        self.BATCH_SIZE = 128
        self.Z_DIM = 100
        self.EPOCHS =1
        self.LOAD_MODEL = False
        self.ADAM_BETA_1 = 0.5
        self.ADAM_BETA_2 = 0.999
        self.LEARNING_RATE = 0.0002
        self.NOISE_PARAM = 0.1

    def preprocess(self,img):
        """
        Normalize and reshape the images
        """
        img = (tf.cast(img, "float32") - 127.5) / 127.5
        return img
    

    def generateImages(self,numSamples=1,imageInputPath="./input",epocas=1,imageOutputPath="./imagenes_sinteticas",lote=16):
        imageInputPath=imageInputPath+"violencia"
        self.BATCH_SIZE=lote
        self.EPOCHS=epocas
        train_data = utils.image_dataset_from_directory(
            imageInputPath,
            labels=None,
            color_mode="rgb",
            image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            seed=42,
            interpolation="bilinear",
        )

    



        train = train_data.map(lambda x: self.preprocess(x))


        train_sample = sample_batch(train)


        discriminator_input = layers.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANNELS))
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(
            discriminator_input
        )
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(
            128, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(
            256, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(
            512, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(
            1,
            kernel_size=4,
            strides=1,
            padding="valid",
            use_bias=False,
            activation="sigmoid",
        )(x)
        discriminator_output = layers.Flatten()(x)

        discriminator = models.Model(discriminator_input, discriminator_output)
        discriminator.summary()


        generator_input = layers.Input(shape=(self.Z_DIM,))
        x = layers.Reshape((1, 1, self.Z_DIM))(generator_input)
        x = layers.Conv2DTranspose(
            512, kernel_size=4, strides=1, padding="valid", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2DTranspose(
            256, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        generator_output = layers.Conv2DTranspose(
            self.CHANNELS,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            activation="tanh",
        )(x)
        generator = models.Model(generator_input, generator_output)
        generator.summary()



        class DCGAN(models.Model):
            def __init__(self, discriminator, generator, latent_dim,NOISE_PARAM):
                super(DCGAN, self).__init__()
                self.discriminator = discriminator
                self.generator = generator
                self.latent_dim = latent_dim
                self.NOISE_PARAM=NOISE_PARAM

            def compile(self, d_optimizer, g_optimizer):
                super(DCGAN, self).compile()
                self.loss_fn = losses.BinaryCrossentropy()
                self.d_optimizer = d_optimizer
                self.g_optimizer = g_optimizer
                self.d_loss_metric = metrics.Mean(name="d_loss")
                self.d_real_acc_metric = metrics.BinaryAccuracy(name="d_real_acc")
                self.d_fake_acc_metric = metrics.BinaryAccuracy(name="d_fake_acc")
                self.d_acc_metric = metrics.BinaryAccuracy(name="d_acc")
                self.g_loss_metric = metrics.Mean(name="g_loss")
                self.g_acc_metric = metrics.BinaryAccuracy(name="g_acc")

            @property
            def metrics(self):
                return [
                    self.d_loss_metric,
                    self.d_real_acc_metric,
                    self.d_fake_acc_metric,
                    self.d_acc_metric,
                    self.g_loss_metric,
                    self.g_acc_metric,
                ]

            def train_step(self, real_images):
                # Sample random points in the latent space
                batch_size = tf.shape(real_images)[0]
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size, self.latent_dim)
                )

                # Train the discriminator on fake images
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = self.generator(
                        random_latent_vectors, training=True
                    )
                    real_predictions = self.discriminator(real_images, training=True)
                    fake_predictions = self.discriminator(
                        generated_images, training=True
                    )

                    real_labels = tf.ones_like(real_predictions)
                    real_noisy_labels = real_labels + self.NOISE_PARAM * tf.random.uniform(
                        tf.shape(real_predictions)
                    )
                    fake_labels = tf.zeros_like(fake_predictions)
                    fake_noisy_labels = fake_labels - self.NOISE_PARAM * tf.random.uniform(
                        tf.shape(fake_predictions)
                    )

                    d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
                    d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
                    d_loss = (d_real_loss + d_fake_loss) / 2.0

                    g_loss = self.loss_fn(real_labels, fake_predictions)

                gradients_of_discriminator = disc_tape.gradient(
                    d_loss, self.discriminator.trainable_variables
                )
                gradients_of_generator = gen_tape.gradient(
                    g_loss, self.generator.trainable_variables
                )

                self.d_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, discriminator.trainable_variables)
                )
                self.g_optimizer.apply_gradients(
                    zip(gradients_of_generator, generator.trainable_variables)
                )

                # Update metrics
                self.d_loss_metric.update_state(d_loss)
                self.d_real_acc_metric.update_state(real_labels, real_predictions)
                self.d_fake_acc_metric.update_state(fake_labels, fake_predictions)
                self.d_acc_metric.update_state(
                    [real_labels, fake_labels], [real_predictions, fake_predictions]
                )
                self.g_loss_metric.update_state(g_loss)
                self.g_acc_metric.update_state(real_labels, fake_predictions)

                return {m.name: m.result() for m in self.metrics}
        

            
            # Create a DCGAN
        dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=self.Z_DIM,NOISE_PARAM=self.NOISE_PARAM)

        if self.LOAD_MODEL:
            dcgan.load_weights("./checkpoint/checkpoint.ckpt")

        dcgan.compile(
            d_optimizer=optimizers.Adam(
                learning_rate=self.LEARNING_RATE, beta_1=self.ADAM_BETA_1, beta_2=self.ADAM_BETA_2
            ),
            g_optimizer=optimizers.Adam(
                learning_rate=self.LEARNING_RATE, beta_1=self.ADAM_BETA_1, beta_2=self.ADAM_BETA_2
            ),
        )        


        # Create a model save checkpoint
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath="./checkpoint/checkpoint.ckpt",
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        )

        tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

        class ImageGenerator(callbacks.Callback):
            def __init__(self, num_img, latent_dim):
                self.num_img = num_img
                self.latent_dim = latent_dim

            def on_epoch_end(self, epoch, logs=None):
                random_latent_vectors = tf.random.normal(
                    shape=(self.num_img, self.latent_dim)
                )
                generated_images = self.model.generator(random_latent_vectors)
                generated_images = generated_images * 127.5 + 127.5
                generated_images = generated_images.numpy()
                display(
                    generated_images,
                    save_to="./output/generated_img_%03d.png" % (epoch),
                )




        dcgan.fit(
            train,
            epochs=self.EPOCHS,
            callbacks=[
                model_checkpoint_callback,
                tensorboard_callback,
                #ImageGenerator(num_img=10, latent_dim=self.Z_DIM),
            ],
        )

        # Save the final models
        generator.save("./models/generator")
        discriminator.save("./models/discriminator")
######################################################################################################################################################################################
        
        ## Sample some points in the latent space, from the standard normal distribution
        noise = np.random.normal(size=( numSamples, self.Z_DIM))
        # Decode the sampled points
        gen_imgs = generator.predict(noise)

        gen_imgs=gen_imgs* 127.5 + 127.564


        gen_imgs=gen_imgs/gen_imgs.max()
        gen_imgs=gen_imgs*255
        gen_imgs=np.array(object=gen_imgs,dtype=np.uint32)


        # Suponiendo que gen_imgs es una lista de imágenes (arrays de NumPy)
        gen_imgsRes = []  # Usamos una lista para almacenar las imágenes redimensionadas

        for img in gen_imgs:
            img = img.astype(np.uint8)  # Asegúrate de que la imagen sea de tipo uint8
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convertir de RGB a BGR
            img = cv2.resize(img, (224, 224))  # Redimensionar la imagen a 224x224
            gen_imgsRes.append(img)  # Agregar la imagen redimensionada a la lista

       
        for img in  gen_imgsRes:
            length_of_string = 16
            cadena="".join(
                    random.choice(string.ascii_letters + string.digits)
                    for _ in range(length_of_string)
                )
            
            cv2.imwrite(imageOutputPath+cadena+'.jpg', img)
            print(">>> Imagen: ",imageOutputPath+cadena+'.jpg'," generada sinteticamente")






######################################################################################################################################################################################











    




    