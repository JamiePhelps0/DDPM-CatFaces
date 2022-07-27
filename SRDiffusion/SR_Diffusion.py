import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import math
from PIL import Image
from tqdm import tqdm


x_T = 1000
batch_size = 4
channels = 64
num_blocks = 2
sr_size = 512
from_size = 64


def diffusion_model(img_size):
    class SelfAttentionModel(tf.keras.Model):
        def __init__(self, input_dims):
            super().__init__()
            self.attn = Attention()
            self.query_conv = tf.keras.layers.Conv2D(input_dims // 8, 1, padding='same')
            self.key_conv = tf.keras.layers.Conv2D(input_dims // 8, 1, padding='same')
            self.value_conv = tf.keras.layers.Conv2D(input_dims, 1, padding='same')

        def call(self, inputs, training=False, mask=None):
            q = self.query_conv(inputs)
            k = self.key_conv(inputs)
            v = self.value_conv(inputs)
            return self.attn([q, k, v, inputs])


    class Attention(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.gamma = None

        def build(self, input_shapes):
            self.gamma = self.add_weight(self.name + '_gamma', shape=(), initializer=tf.initializers.Zeros)

        def call(self, inputs, training=False, mask=None):
            query_tensor = inputs[0]
            key_tensor = inputs[1]
            value_tensor = inputs[2]
            origin_input = inputs[3]

            batch, height, width, _ = inputs[0].shape

            proj_query = tf.reshape(query_tensor, (batch, height*width, -1))
            proj_key = tf.transpose(tf.reshape(key_tensor, (batch, height*width, -1)), (0, 2, 1))
            proj_value = tf.transpose(tf.reshape(value_tensor, (batch, height*width, -1)), (0, 2, 1))

            energy = tf.matmul(proj_query, proj_key)
            attention = tf.nn.softmax(energy)
            x_out = tf.matmul(proj_value, tf.transpose(attention, (0, 2, 1)))

            x_out = tf.reshape(tf.transpose(x_out, (0, 2, 1)), (batch, height, width, -1))

            return tf.add(tf.multiply(x_out, self.gamma), origin_input)

    class ResNetBlock(tf.keras.Model):
        def __init__(self, num_channels):
            super().__init__()

            self._channels = num_channels

            self.conv_1 = tf.keras.layers.Conv2D(self._channels, 3, padding='same')
            self.conv_2 = tf.keras.layers.Conv2D(self._channels, 3, padding='same')
            self.conv_res = tf.keras.layers.Conv2D(self._channels, 1, padding='same')
            self.swish = tf.keras.activations.swish
            self.norm_1 = tfa.layers.GroupNormalization(groups=8)
            self.norm_2 = tfa.layers.GroupNormalization(groups=8)
            self.add = tf.keras.layers.Add()

        def call(self, inputs, training=None, mask=None):
            _, height, width, _ = inputs.shape
            res = inputs

            y = self.norm_1(inputs)
            y = self.swish(y)
            y = self.conv_1(y)
            y = self.norm_2(y)
            y = self.swish(y)
            y = self.conv_2(y)

            res = self.conv_res(res)

            y = self.add([y, res])
            return y

        def get_config(self):
            pass

    class ResBlock(tf.keras.Model):
        def __init__(self, num_channels, num_res_blocks):
            super().__init__()

            self._channels = num_channels
            self._res_blocks = num_res_blocks

            self.res_blocks = [ResNetBlock(self._channels) for _ in range(self._res_blocks)]

        def call(self, inputs, training=None, mask=None):
            y = inputs
            for block in self.res_blocks:
                y = block(y)
            return y

        def get_config(self):
            pass

    def up_conv(channel):
        return tf.keras.layers.Conv2DTranspose(channel, 3, 2, padding='same')

    def down_conv(channel):
        return tf.keras.layers.Conv2D(channel, 3, 2, padding='same')

    def get_timestep_embedding(timesteps, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float64) * -emb)
        emb = tf.cast(timesteps, dtype=tf.float64)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
        if embedding_dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb

    x_t_in = tf.keras.layers.Input(batch_shape=(batch_size, img_size, img_size, 3))
    t_emb_in = tf.keras.layers.Input(batch_shape=[batch_size])
    x_e_in = tf.keras.layers.Input(batch_shape=(batch_size, img_size, img_size, channels))

    one_on_root_2 = tf.constant(1 / np.sqrt(2), dtype=tf.float32)

    # t_emb_in = tf.keras.layers.Reshape([batch_size])(t_emb_in)
    t_emb = get_timestep_embedding(t_emb_in, channels)

    t_emb = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)
    t_emb = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)

    out = tf.keras.layers.Conv2D(channels, 3, padding='same')(x_t_in)
    out = tf.keras.layers.Add()([out, x_e_in])

    out = down_conv(channels)(out)
    t_emb_dense = tf.keras.layers.Dense(channels, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels, num_blocks)(out)
    res_1 = out * one_on_root_2

    out = down_conv(channels * 2)(out)
    t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels * 2, num_blocks)(out)
    res_2 = out * one_on_root_2

    out = down_conv(channels * 2)(out)
    t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels * 2, num_blocks * 2)(out)
    out = SelfAttentionModel(channels * 2)(out)
    res_3 = out * one_on_root_2

    out = down_conv(channels * 4)(out)
    t_emb_dense = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels * 4, num_blocks * 2)(out)
    out = SelfAttentionModel(channels * 4)(out)
    res_4 = out * one_on_root_2

    out = ResBlock(channels * 4, num_blocks * 3)(out)

    out = tf.keras.layers.Add()([out, res_4])
    t_emb_dense = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels * 4, num_blocks * 2)(out)
    out = SelfAttentionModel(channels * 4)(out)
    out = up_conv(channels * 2)(out)

    out = tf.keras.layers.Add()([out, res_3])
    t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels * 2, num_blocks * 2)(out)
    out = SelfAttentionModel(channels * 2)(out)
    out = up_conv(channels * 2)(out)

    out = tf.keras.layers.Add()([out, res_2])
    t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels * 2, num_blocks)(out)
    out = up_conv(channels)(out)

    out = tf.keras.layers.Add()([out, res_1])
    t_emb_dense = tf.keras.layers.Dense(channels, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
    out = tf.keras.layers.Add()([out, t_emb_dense])
    out = ResBlock(channels, num_blocks)(out)
    out = up_conv(channels)(out)

    out = tf.keras.layers.Conv2D(channels, 3, padding='same')(out)
    out = tf.keras.activations.swish(out)
    out = tf.keras.layers.Conv2D(3, 3, padding='same')(out)
    return tf.keras.models.Model(inputs=[x_t_in, t_emb_in, x_e_in], outputs=out)


def lr_encoder(input_size):
    tf.keras.backend.set_floatx('float32')

    def res_block(ch=128, k_s=3, st=1):
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
        x = tf.nn.depth_to_space(x, 2)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    input_lr = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    input_conv = tf.keras.layers.Conv2D(128, 9, padding='same')(input_lr)
    input_conv = tf.keras.layers.LeakyReLU()(input_conv)
    x = input_conv
    for x in range(5):
        res_output = res_block()(x)
        x = tf.keras.layers.Add()([x, res_output])
    x = tf.keras.layers.Conv2D(128, 9, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, input_conv])
    x = upsample_block(x)
    x = upsample_block(x)
    x = upsample_block(x)
    x = tf.keras.layers.Conv2D(3, 9, activation='tanh', padding='same')(x)
    SRResnet = tf.keras.models.Model(input_lr, x)
    SRResnet.load_weights(f'LR_ENC/SR_64x512_ResNet_W.h5')
    return SRResnet


SR_Resnet = lr_encoder(from_size)


# Define models and helper functions


def rescale(normalized_img):
    normalized_img = tf.cast((normalized_img + 1) * 127.5, tf.int32)
    return normalized_img


model = diffusion_model(sr_size)
print(model.summary())

lr_encoder = tf.keras.models.Model(inputs=SR_Resnet.input, outputs=SR_Resnet.get_layer('leaky_re_lu_13').output)
print(lr_encoder.summary())


def tf_func(y):
    y = tf.reshape(y, (sr_size, sr_size, 3))   # Reshape input data from (1, 256, 256, 3) to (256, 256, 3)
    y = tf.cast((y - 127.5) / 127.5, dtype=tf.float64)   # Normalize from [0, 255] to [-1, 1]
    return y


def get_dataset():
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        'E:/ML Data/afhq/train/cat/', labels=None, label_mode=None,
        batch_size=1, image_size=(sr_size, sr_size))
    ds = ds.map(tf_func)
    ds = ds.batch(batch_size, drop_remainder=True).shuffle(64).prefetch(1)
    return ds


def train_epoch(dataset, sqrt_alphas, model, om_sqrt_alphas, optimizer):
    losses = []
    for batch in tqdm(dataset):
        x_low = tf.image.resize(batch, (from_size, from_size), method=tf.image.ResizeMethod.BICUBIC)
        x_up = tf.image.resize(x_low, (sr_size, sr_size), method=tf.image.ResizeMethod.BICUBIC)
        x_r = batch - tf.cast(x_up, dtype=tf.float64)
        x_e = lr_encoder(x_low)

        t = tf.random.uniform([batch_size], minval=1, maxval=x_T, dtype=tf.int32)
        sqrt_a = tf.reshape(tf.gather(sqrt_alphas, t.numpy()), (batch_size, 1, 1, 1))
        om_sqrt_a = tf.reshape(tf.gather(om_sqrt_alphas, t.numpy()), (batch_size, 1, 1, 1))
        eps = tf.random.normal([batch_size, sr_size, sr_size, 3], dtype=tf.float64)
        x = tf.cast(sqrt_a * x_r + om_sqrt_a * eps, dtype=tf.float64)

        with tf.GradientTape() as tape:
            eps_pred = tf.cast(model([x, t, x_e]), dtype=tf.float64)
            loss = tf.reduce_mean(tf.math.squared_difference(eps, eps_pred))

        losses.append(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(np.mean(losses))
    model.save_weights('SR_saves/SR_512_Diff.h5')


dataset = get_dataset()

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

betas = np.linspace(0.0001, 0.02, num=x_T)

alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
betas = tf.constant(betas, dtype=tf.float64)
alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float64)
sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float64)
sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1. - alphas_cumprod), dtype=tf.float64)
om_a_on_sqrt_om_acumprod = tf.constant((1 - alphas) / tf.sqrt(1 - alphas_cumprod), dtype=tf.float64)
one_on_sqrt_alpha = tf.constant(1 / tf.sqrt(alphas), dtype=tf.float64)
alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
sigma = tf.constant(tf.sqrt(betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)), dtype=tf.float64)


def sample(shape, denoise_fn, lr_image_batch):
    batch, height, width, num_channels = shape
    lr_up = tf.image.resize(lr_image_batch, (sr_size, sr_size), method=tf.image.ResizeMethod.BICUBIC).numpy()
    x_e = lr_encoder(lr_image_batch)
    x_t = tf.random.normal(shape=shape, dtype=tf.float64)
    for t in tqdm(reversed(range(1, x_T + 1)), total=1000):
        if t > 1:
            z = tf.random.normal(shape=shape, dtype=tf.float64)
        else:
            z = tf.zeros(shape=shape, dtype=tf.float64)
        t_batch = tf.constant(t, shape=batch)
        eps_pred = tf.cast(denoise_fn([x_t, t_batch, x_e]), tf.float64)
        x_t -= tf.reshape(tf.gather(om_a_on_sqrt_om_acumprod, t_batch.numpy()), (batch, 1, 1, 1)) * eps_pred
        x_t *= tf.reshape(tf.gather(one_on_sqrt_alpha, t_batch.numpy()), (batch, 1, 1, 1))
        x_t += tf.reshape(tf.gather(sigma, t_batch.numpy()), (batch, 1, 1, 1)) * z
    return x_t, lr_up


model.load_weights('SR_saves/SR_512_Diff.h5')


for i in range(50):
    train_epoch(dataset, sqrt_alphas_cumprod, model, sqrt_one_minus_alphas_cumprod, opt)

# model.save_weights('SR_diffusion.save', overwrite=True)

images = []
for i in range(batch_size):
    images.append(np.array(Image.open('C:/Users/Jamie Phelps/Pictures/FakeCat/cat_{0}.png'.format(i + 13))))

lr_images = tf.constant(images, dtype=tf.float64)
lr_images = tf.cast((lr_images - 127.5) / 127.5, dtype=tf.float64)

img, lr_upsample = sample((batch_size, sr_size, sr_size, 3), model, lr_images)

img = np.split(img, batch_size)

img_new = np.copy(img)

for i, image in enumerate(img_new):
    image += lr_upsample[i]
    image = rescale(image)
    image = np.clip(image, 0, 255)
    image = Image.fromarray(np.array(image, dtype=np.uint8).reshape((sr_size, sr_size, 3)))
    image.save('C:/Users/Jamie Phelps/Pictures/FakeCat/cat_{0}.png'.format(i))


