import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math
from tqdm import tqdm
from PIL import Image


x_T = 1000
batch_size = 32
channels = 96
num_blocks = 2
opt = tf.keras.optimizers.Adam(learning_rate=0.000007)


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
    def __init__(self, channels):
        super().__init__()

        self._channels = channels

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

        x = self.norm_1(inputs)
        x = self.swish(x)
        x = self.conv_1(x)
        x = self.norm_2(x)
        x = self.swish(x)
        x = self.conv_2(x)

        res = self.conv_res(res)

        x = self.add([x, res])
        return x


class ResBlock(tf.keras.Model):
    def __init__(self, channels, num_res_blocks):
        super().__init__()

        self._channels = channels
        self._res_blocks = num_res_blocks

        self.res_blocks = [ResNetBlock(self._channels) for _ in range(self._res_blocks)]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for block in self.res_blocks:
            x = block(x)
        return x


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


x_t_in = tf.keras.layers.Input(batch_shape=(batch_size, 64, 64, 3))
t_emb_in = tf.keras.layers.Input(batch_shape=[batch_size])

one_on_root_2 = tf.constant(1 / np.sqrt(2), dtype=tf.float32)

t_emb = get_timestep_embedding(t_emb_in, channels)

t_emb = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)
t_emb = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)

x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x_t_in)

x = down_conv(channels)(x)
t_emb_dense = tf.keras.layers.Dense(channels, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels, num_blocks)(x)
res_1 = x * one_on_root_2

x = down_conv(channels * 2)(x)
t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels * 2, num_blocks)(x)
res_2 = x * one_on_root_2

x = down_conv(channels * 2)(x)
t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels * 2, num_blocks * 2)(x)
x = SelfAttentionModel(channels * 2)(x)
res_3 = x * one_on_root_2

x = down_conv(channels * 4)(x)
t_emb_dense = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels * 4, num_blocks * 2)(x)
x = SelfAttentionModel(channels * 4)(x)
res_4 = x * one_on_root_2

x = ResBlock(channels * 4, num_blocks * 3)(x)

x = tf.keras.layers.Add()([x, res_4])
t_emb_dense = tf.keras.layers.Dense(channels * 4, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels * 4, num_blocks * 2)(x)
x = SelfAttentionModel(channels * 4)(x)
x = up_conv(channels * 2)(x)

x = tf.keras.layers.Add()([x, res_3])
t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels * 2, num_blocks * 2)(x)
x = SelfAttentionModel(channels * 2)(x)
x = up_conv(channels * 2)(x)

x = tf.keras.layers.Add()([x, res_2])
t_emb_dense = tf.keras.layers.Dense(channels * 2, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels * 2, num_blocks)(x)
x = up_conv(channels)(x)

x = tf.keras.layers.Add()([x, res_1])
t_emb_dense = tf.keras.layers.Dense(channels, activation=tf.keras.activations.swish)(t_emb)[:, None, None, :]
x = tf.keras.layers.Add()([x, t_emb_dense])
x = ResBlock(channels, num_blocks)(x)
x = up_conv(channels)(x)

x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)
x = tf.keras.activations.swish(x)
out = tf.keras.layers.Conv2D(3, 3, padding='same')(x)


def rescale(normalized_img):
    normalized_img = tf.cast((normalized_img + 1) * 127.5, tf.int32)
    return normalized_img


model = tf.keras.models.Model(inputs=[x_t_in, t_emb_in], outputs=out)
print(model.summary())


def tf_func(y):
    y = tf.reshape(y, (64, 64, 3))
    y = tf.cast((y - 127.5) / 127.5, dtype=tf.float64)
    return y


def get_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Jamie Phelps/Documents/Cats/Cat-faces-dataset-master/cat_train/', labels=None, label_mode=None,
        batch_size=1, image_size=(64, 64))
    dataset = dataset.map(tf_func)
    dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(1024).prefetch(3)
    return dataset


def train_epoch(dataset, sqrt_alphas, model, om_sqrt_alphas, optimizer):
    losses = []
    for batch in tqdm(dataset):
        t = tf.random.uniform([batch_size], minval=1, maxval=x_T, dtype=tf.int32)
        sqrt_a = tf.reshape(tf.gather(sqrt_alphas, t.numpy()), (batch_size, 1, 1, 1))
        om_sqrt_a = tf.reshape(tf.gather(om_sqrt_alphas, t.numpy()), (batch_size, 1, 1, 1))
        eps = tf.random.normal([batch_size, 64, 64, 3], dtype=tf.float64)
        # print(sqrt_a.shape, batch.shape, )
        x = tf.cast(sqrt_a * batch + om_sqrt_a * eps, dtype=tf.float64)
        # print(t_in.dtype, x.shape)
        with tf.GradientTape() as tape:
            eps_pred = tf.cast(model([x, t]), dtype=tf.float64)
            # delta_eps = eps - tf.cast(eps_pred, dtype=tf.float64)
            # print(delta_eps)
            loss = tf.reduce_mean(tf.math.squared_difference(eps, eps_pred))

        losses.append(loss)
        # print(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(np.mean(losses))


dataset = get_dataset()

betas = np.linspace(0.0001, 0.02, num=x_T)

alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
betas = tf.constant(betas, dtype=tf.float64)
alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float64)
sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float64)
sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1. - alphas_cumprod), dtype=tf.float64)
om_a_on_sqrt_om_acumprod = tf.constant((1 - alphas) / tf.sqrt(1 - alphas_cumprod))
one_on_sqrt_alpha = tf.constant(1 / tf.sqrt(alphas))
sigma = tf.constant(tf.sqrt(betas))


def sample(shape, denoise_fn):
    batch, height, width, num_channels = shape
    x_t = tf.random.normal(shape=shape, dtype=tf.float64)
    for t in tqdm(reversed(range(1, x_T + 1)), total=1000):
        if t != 1:
            z = tf.random.normal(shape=shape, dtype=tf.float64)
        else:
            z = tf.zeros(shape=shape, dtype=tf.float64)
        t_batch = tf.constant(t, shape=batch)
        eps_pred = tf.cast(denoise_fn([x_t, t_batch]), tf.float64)
        x_t -= tf.reshape(tf.gather(om_a_on_sqrt_om_acumprod, t_batch.numpy()), (batch, 1, 1, 1)) * eps_pred
        x_t *= tf.reshape(tf.gather(one_on_sqrt_alpha, t_batch.numpy()), (batch, 1, 1, 1))
        x_t += tf.reshape(tf.gather(sigma, t_batch.numpy()), (batch, 1, 1, 1)) * z
    return x_t


model.load_weights('diffusion2.h5')


# for i in range(50):
#     train_epoch(dataset, sqrt_alphas_cumprod, model, sqrt_one_minus_alphas_cumprod, opt)

model.save_weights('diffusion.h5', overwrite=True)
img = sample((64, 64, 64, 3), model)

img = np.split(img, 64)
img = rescale(img)
img = np.clip(img, 0, 255)


for i, image in enumerate(img):
    image = Image.fromarray(np.array(image, dtype=np.uint8).reshape((64, 64, 3)))
    image.save('C:/Users/Jamie Phelps/Pictures/FakeCat/cat_{0}.png'.format(i))
