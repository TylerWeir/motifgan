import librosa
import numpy as np
import soundfile
from IPython.display import Audio
import librosa.display
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import librosa
import soundfile as sf


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):

            sample = generated_images[i]

            # Save the numpy array
            np.save("output-arrays/generated_%03d_%d.npy" % (epoch, i), sample)
            #write wav
            des = np.zeros([128, 312], dtype=np.complex64)

            for i in range(len(sample)):
                for k in range(len(sample[0])):
                    des[i][k] = complex(sample[i][k][0], sample[i][k][1])
            res = librosa.istft(des)
            # Save a spectrogram

            des = np.zeros([128, 312], dtype=np.complex64)

            for i in range(len(sample)):
                for k in range(len(sample[0])):
                    des[i][k] = complex(sample[i][k][0], sample[i][k][1])

            res = librosa.istft(des)
            sf.write("output-wav/generated_%03d_%d.wav" % (epoch, i), res, 4096)

            # convert the slices to amplitude
            sgram_db = librosa.amplitude_to_db(abs(des))

            _, ax = plt.subplots(figsize=(5, 5))

            librosa.display.specshow(sgram_db, sr=4096, x_axis='time', y_axis='log', ax=ax, cmap='gray')

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0)

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, )
            plt.margins(0, 0)
            plt.savefig("output-specs/generated_%03d_%d.png" % (epoch, i))
            plt.close()


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


# Press the green button in the gutter to run the script.
def play_plot():
    try:
        sample = np.load("output-arrays\generated_004_0.npy")
    except:
        "failure to load"
    des = np.zeros([128, 312], dtype=np.complex64)

    for i in range(len(sample)):
        for k in range(len(sample[0])):
            des[i][k] = complex(sample[i][k][0], sample[i][k][1])

    res = librosa.istft(des)

    # convert the slices to amplitude
    sgram_db = librosa.amplitude_to_db(abs(des))

    _, ax = plt.subplots(figsize=(5, 5))

    librosa.display.specshow(sgram_db, sr=4096, x_axis='time', ax=ax, cmap='gray')

    try:
        # We'll need IPython.display's Audio widget
        from IPython.display import Audio
        Audio(data=res, rate=4096)
    except:
        print("Failure to print data")


def convert_audio_to_complex_array(filename, outfilename=None, overwrite=False, vertical_res=256):
    """convert_audio_to_complex_array -- using librosa's short time Fourier transform.

    Arguments:
    filename -- filepath to the file that you to copy to an array
    outfilename -- filepath to the output array
    overwrite -- whether to overwrite if a file already exists with the given outfilename
    vertical resolution -- put this in or 256 is default
    sample rate = 4096 default but can be increased
    Returns -- None
    """

    # sr == sampling rate
    audio_data, sr = librosa.load(filename, sr=4096)

    # Apply the short time Fourier transform
    result = librosa.stft(audio_data, center=False, n_fft=vertical_res, win_length=vertical_res)
    np.save(filename[:-4] + ".npy", result)


def test():
    ## KERAS MODELS
    latent_dim = 128

    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(16 * 39 * 128),
            layers.Reshape((16, 39, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(2, kernel_size=5, padding="same", activation="tanh"),
        ],
        name="generator",
    )
    generator.summary()
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(128, 312, 2)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    discriminator.summary()
    # TODO MAYBE HERE? SKIP
    # Load dataset from directory with keras
    mega_tensor = np.load("data.npy")

    train_ds = tf.data.Dataset.from_tensor_slices(mega_tensor)
    dataset = train_ds.batch(8)

    # TODO MAKE THIS ARG?
    epochs = 300  # In practice, use ~100 epochs

    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=128)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    gan.fit(
        dataset, epochs=epochs, callbacks=[GANMonitor(num_img=1, latent_dim=128)]
    )


if __name__ == '__main__':
    # Check that TensorFlow can see the GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)

    # TODO MAKE/CHECK ALL DIR
    if os.path.isdir("samples"):
        print("Sample Directory OK")
        pass
    else:
        os.mkdir("samples")
    if os.path.isdir("output-arrays"):
        print("output-arrays Directory OK")
    else:
        os.mkdir("output-arrays")
    if os.path.isdir("output-specs"):
        print("output-specs Directory OK")
    else:
        os.mkdir("output-specs")
    if os.path.isdir("output-wav"):
        print("output-wav Directory OK")
    else:
        os.mkdir("output-wav")

    # TODO SET UP PORTION MAKE ABILITY TO SKIP
    # Convert all the files to numpy arrays and save
    user = input("Do you have a megaTensor already? (Y/N) ")
    if user.lower() == "y":
        test()
    else:

        for i, item in enumerate(os.listdir("samples/")):
            convert_audio_to_complex_array("samples/" + item)
        # Delete the mp3 files
        for i, item in enumerate(os.listdir("samples/")):
            if item.endswith(".mp3"):
                os.remove("samples/" + item)

        target_len = 312
        try:
            target_height = len(np.load("samples/" + os.listdir("samples/")[0])) - 1
        except:
            print("No Samples Please Recheck Directory")
            exit(0)
        target_samples = len(os.listdir("samples/"))
        channels = 2

        print(target_height)

        mega_tensor = np.zeros([target_samples, target_height, target_len, channels], dtype=np.float32)

        # Add every sample to the mega tensor
        for i, name in enumerate(os.listdir("samples/")):
            try:
                item = np.load("samples/" + name)

                for j in range(len(item) - 1):
                    for k in range(len(item[0])):
                        if k < target_len:
                            mega_tensor[i][j][k][0] = np.real(item[j][k])
                            mega_tensor[i][j][k][1] = np.imag(item[j][k])
            except:
                print("Error In making mega tensor")
                exit(0)
        # Then save the mega tensor
        np.save("data.npy")

        # TODO SKIP TO HERE

        ## KERAS MODELS
        latent_dim = 128

        generator = keras.Sequential(
            [
                keras.Input(shape=(latent_dim,)),
                layers.Dense(16 * 39 * 128),
                layers.Reshape((16, 39, 128)),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(2, kernel_size=5, padding="same", activation="tanh"),
            ],
            name="generator",
        )
        generator.summary()
        discriminator = keras.Sequential(
            [
                keras.Input(shape=(128, 312, 2)),
                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        discriminator.summary()
        # TODO MAYBE HERE? SKIP
        # Load dataset from directory with keras
        mega_tensor = np.load("data.npy")

        train_ds = tf.data.Dataset.from_tensor_slices(mega_tensor)
        dataset = train_ds.batch(8)

        # TODO MAKE THIS ARG?
        epochs = 300  # In practice, use ~100 epochs

        gan = GAN(discriminator=discriminator, generator=generator, latent_dim=128)
        gan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss_fn=keras.losses.BinaryCrossentropy(),
        )
        gan.fit(
            dataset, epochs=epochs, callbacks=[GANMonitor(num_img=1, latent_dim=128)]
        )
