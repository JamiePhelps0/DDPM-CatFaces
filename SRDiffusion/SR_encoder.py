import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

batch_size = 4
learning_rate = 0.00003
image_in = 64
image_out = 512


def tf_func(y):
    y = tf.reshape(y, (image_out, image_out, 3))
    x = tf.cast((y - 127.5) / 127.5, dtype=tf.float64)
    y = tf.image.resize(x, (image_in, image_in), method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(y, dtype=tf.float64), x


def get_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Jamie Phelps/Documents/Cats/afhq/train/cat', labels=None,
        batch_size=1, image_size=(image_out, image_out))
    dataset = dataset.map(tf_func)
    dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(64).prefetch(1)
    return dataset


def lr_encoder(input_size):
    tf.keras.backend.set_floatx('float32')

    def residual_block_gen(ch=128, k_s=3, st=1):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])
        return model

    def upsample_block(x, ch=256, k_s=3, st=1):
        x = tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same')(x)
        x = tf.nn.depth_to_space(x, 2)  # Subpixel pixelshuffler
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    input_lr = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    input_conv = tf.keras.layers.Conv2D(128, 9, padding='same')(input_lr)
    input_conv = tf.keras.layers.LeakyReLU()(input_conv)
    SRRes = input_conv
    for x in range(5):
        res_output = residual_block_gen()(SRRes)
        SRRes = tf.keras.layers.Add()([SRRes, res_output])
    SRRes = tf.keras.layers.Conv2D(128, 9, padding='same')(SRRes)
    SRRes = tf.keras.layers.BatchNormalization()(SRRes)
    SRRes = tf.keras.layers.Add()([SRRes, input_conv])
    SRRes = upsample_block(SRRes)
    SRRes = upsample_block(SRRes)
    SRRes = upsample_block(SRRes)
    SRRes = tf.keras.layers.Conv2D(3, 9, activation='tanh', padding='same')(SRRes)
    SRResnet = tf.keras.models.Model(input_lr, SRRes)
    # SRResnet.load_weights(f'LR_ENC/SR_512_ResNet_W.h5')
    return SRResnet


SRResnet = lr_encoder(image_in)


def pixel_MSE(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


@tf.function
def train_step(data):
    low_resolution, high_resolution = data
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        super_resolution = tf.cast(SRResnet(low_resolution, training=True), tf.float64)
        loss = pixel_MSE(high_resolution, super_resolution)
    gradients_of_generator = gen_tape.gradient(loss, SRResnet.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_generator, SRResnet.trainable_variables))
    return loss


SRResnet.load_weights('LR_ENC/SR_64x512_ResNet_W.h5')
dataset = get_dataset()

for x in range(500):
    losses = []
    for image_batch in tqdm(dataset, position=0, leave=True):
        loss = train_step(image_batch)
        losses.append(loss)
    print(np.mean(losses))
    SRResnet.save_weights(f'LR_ENC/SR_{image_in}x{image_out}_ResNet_W.h5')


def rescale(normalized_img):
    normalized_img = tf.cast((normalized_img + 1) * 127.5, tf.int32)
    return normalized_img


images = []
for i in range(batch_size):
    images.append(np.array(Image.open('C:/Users/Jamie Phelps/Pictures/FakeCat/cat_{0}.png'.format(i + 13))))

images = tf.constant(images, dtype=tf.float64)
images = tf.cast((images - 127.5) / 127.5, dtype=tf.float64)

sr = SRResnet(images)
sr = rescale(sr)
for i, image in enumerate(sr):
    image = np.clip(image, 0, 255)
    image = Image.fromarray(np.array(image, dtype=np.uint8).reshape((image_out, image_out, 3)))
    image.save('C:/Users/Jamie Phelps/Pictures/FakeCat/cat_{0}.png'.format(i))

