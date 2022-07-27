import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math
from tqdm import tqdm
from PIL import Image


x_T = 1000
batch_size = 24
channels = 128
num_blocks = 3
c_drop = 0.1
guidance = 1
img_size = 128


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

        input_shape = tf.shape(query_tensor)

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


x_t_in = tf.keras.layers.Input(batch_shape=(batch_size, img_size, img_size, 6))
t_emb_in = tf.keras.layers.Input(batch_shape=[batch_size])

one_on_root_2 = tf.constant(1 / np.sqrt(2), dtype=tf.float32)

# t_emb_in = tf.keras.layers.Reshape([batch_size])(t_emb_in)
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


def tf_func(y, label):
    y = tf.reshape(y, (img_size, img_size, 3))
    y = tf.cast((y - 127.5) / 127.5, dtype=tf.float64)
    label = tf.expand_dims(label, axis=1)
    label = tf.expand_dims(label, axis=1)
    label = tf.tile(label, (1, img_size, img_size, 1))
    return y, tf.cast(label, dtype=tf.float64)


def get_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Jamie Phelps/Documents/Cats/afhq/train', labels='inferred', label_mode='categorical',
        batch_size=1, image_size=(img_size, img_size))
    dataset = dataset.map(tf_func)
    dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(64).prefetch(1)
    return dataset


def train_epoch(dataset, sqrt_alphas, model, om_sqrt_alphas, optimizer):
    losses = []
    for batch in tqdm(dataset, total=len(dataset), colour='white'):
        x, y = batch
        t = tf.random.uniform([batch_size], minval=1, maxval=x_T, dtype=tf.int32)
        sqrt_a = tf.reshape(tf.gather(sqrt_alphas, t.numpy()), (batch_size, 1, 1, 1))
        om_sqrt_a = tf.reshape(tf.gather(om_sqrt_alphas, t.numpy()), (batch_size, 1, 1, 1))
        eps = tf.random.normal((batch_size, img_size, img_size, 3), dtype=tf.float64)
        x = tf.cast(sqrt_a * x + om_sqrt_a * eps, dtype=tf.float64)
        if np.random.uniform(0, 1.0) < c_drop:
            y = np.zeros(y.shape, dtype=np.float64)
        y = tf.reshape(y, (batch_size, img_size, img_size, 3))
        x = tf.concat([x, y], axis=-1)

        with tf.GradientTape() as tape:
            eps_pred = tf.cast(model([x, t]), dtype=tf.float64)
            loss = tf.reduce_mean(tf.math.squared_difference(eps, eps_pred))

        losses.append(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(np.mean(losses))
    model.save_weights(f'diff/HQdiffusion{img_size}.h5', overwrite=True)


dataset = get_dataset()

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

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


def sample(shape, denoise_fn, condition=None):
    batch, height, width, num_channels = shape
    x_t = tf.random.normal(shape=shape, dtype=tf.float64)
    zeros = tf.zeros(shape, dtype=tf.float64)

    def broadcast_classes(inputs):
        one_hot = np.zeros((inputs.size, 3), dtype=np.int)
        one_hot[np.arange(inputs.size), inputs] = 1
        inputs = tf.constant(one_hot, dtype=tf.int32)
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.tile(inputs, (1, img_size, img_size, 1))
        return tf.cast(inputs, tf.float64)

    if condition is None:
        classes = np.random.randint(0, 3, [batch])
        classes = broadcast_classes(classes)
    else:
        classes = np.tile(condition, batch)
        classes = broadcast_classes(classes)

    for t in tqdm(reversed(range(1, x_T + 1)), total=1000):
        if t != 1:
            z = tf.random.normal(shape=shape, dtype=tf.float64)
        else:
            z = tf.zeros(shape=shape, dtype=tf.float64)
        t_batch = tf.constant(t, shape=batch)
        x_c_input = tf.concat([x_t, classes], axis=-1)
        x_input = tf.concat([x_t, zeros], axis=-1)
        eps_c = tf.cast(denoise_fn([x_c_input, t_batch]), tf.float64)
        eps = tf.cast(denoise_fn([x_input, t_batch]), tf.float64)
        # eps_c = tf.clip_by_value(eps_c, -1, 1)
        # eps = tf.clip_by_value(eps, -1, 1)
        eps_pred = (1 + guidance) * eps_c - guidance * eps
        x_t -= tf.reshape(tf.gather(om_a_on_sqrt_om_acumprod, t_batch.numpy()), (batch, 1, 1, 1)) * eps_pred
        x_t *= tf.reshape(tf.gather(one_on_sqrt_alpha, t_batch.numpy()), (batch, 1, 1, 1))
        x_t += tf.reshape(tf.gather(sigma, t_batch.numpy()), (batch, 1, 1, 1)) * z
    return x_t


model.load_weights('diff/HQdiffusion128.h5')

# for i in range(600):
#     train_epoch(dataset, sqrt_alphas_cumprod, model, sqrt_one_minus_alphas_cumprod, opt)


img = sample((batch_size, img_size, img_size, 3), model, condition=0)

img = np.split(img, batch_size)

print(img)
for i, image in enumerate(img):
    imag = rescale(image)
    imag = np.clip(imag, 0, 255)
    imag = Image.fromarray(np.array(imag, dtype=np.uint8).reshape((img_size, img_size, 3)))
    imag.save('C:/Users/Jamie Phelps/Pictures/FakeCat/cat_{0}.png'.format(i))
