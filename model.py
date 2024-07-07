import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import AdamW

def channel_attention_module(x, ratio=8):
    batch, _, _, channel = x.shape
    ## Shared layers
    l1 = keras.layers.Dense(channel//ratio, activation="relu", use_bias=False)
    l2 = keras.layers.Dense(channel, use_bias=False)
    ## Global Average Pooling
    x1 = keras.layers.GlobalAveragePooling2D()(x)
    x1 = l1(x1)
    x1 = l2(x1)
    ## Global Max Pooling
    x2 = keras.layers.GlobalMaxPooling2D()(x)
    x2 = l1(x2)
    x2 = l2(x2)
    ## Add both the features and pass through sigmoid
    feats = x1 + x2
    feats = keras.layers.Activation("sigmoid")(feats)
    feats = keras.layers.Multiply()([x, feats])
    return feats

def spatial_attention_module(x):
    ## Average Pooling
    x1 = tf.reduce_mean(x, axis=-1)
    x1 = tf.expand_dims(x1, axis=-1)
    ## Max Pooling
    x2 = tf.reduce_max(x, axis=-1)
    x2 = tf.expand_dims(x2, axis=-1)
    ## Concatenat both the features
    feats = keras.layers.Concatenate()([x1, x2])
    ## Conv layer
    feats = keras.layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(feats)
    feats = keras.layers.Multiply()([x, feats])

    return feats

def cbam(x):
    x = channel_attention_module(x)
    x = spatial_attention_module(x)
    return x

class pam(tf.keras.layers.Layer):
    def __init__(self, in_dim):
        super(pam, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2D(in_dim // 8, kernel_size=1)
        self.key_conv = Conv2D(in_dim // 8, kernel_size=1)
        self.value_conv = Conv2D(in_dim, kernel_size=1)
        self.gamma = tf.Variable(tf.zeros((1,)), trainable=True)

        self.softmax = Softmax(axis=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (B X H X W X C)
        returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
        """
        m_batchsize, height, width, C = x.shape
        proj_query = tf.transpose(tf.reshape(self.query_conv(x), [m_batchsize, height * width, -1]), perm=[0, 2, 1])
        proj_key = tf.reshape(self.key_conv(x), [m_batchsize, -1, height * width])
        energy = tf.matmul(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = tf.reshape(self.value_conv(x), [m_batchsize, -1, height * width])
        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, [m_batchsize, height, width, -1])
        out = self.gamma * out + x
        return out
    
def pcbam(x):
    x_c = channel_attention_module(x)
    x_s = spatial_attention_module(x_c)
    x_p = pam(24)(x)
    out = x_s + x_p
    return out

class SWA(tf.keras.layers.Layer):
    def __init__(self, in_dim, n_heads=8, window_size=7):
        super(SWA, self).__init__()
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.window_size = window_size

        self.query_conv = Conv2D(in_dim, kernel_size=1)
        self.key_conv = Conv2D(in_dim, kernel_size=1)
        self.value_conv = Conv2D(in_dim, kernel_size=1)
        self.gamma = tf.Variable(tf.zeros((1,)), trainable=True)

        self.softmax = Softmax(axis=-1)

    def forward(self, x):
        """
        inputs:
            x: input feature maps (B X H X W X C)
        returns:
            out: attention value + input feature
        """
        m_batchsize, height, width, C = x.shape
        padded_x = tf.pad(x, [[0, 0], [self.window_size // 2, self.window_size // 2], [self.window_size // 2, self.window_size // 2], [0, 0]], "REFLECT")

        # Reshape queries, keys, and values
        proj_query = tf.transpose(self.query_conv(x), perm=[0, 3, 1, 2])  # (B X C X H X W)
        proj_key = tf.transpose(self.key_conv(padded_x), perm=[0, 3, 1, 2])  # (B X C X H X W)
        proj_value = tf.transpose(self.value_conv(padded_x), perm=[0, 3, 1, 2])  # (B X C X H X W)

        # Reshape window
        window = tf.image.extract_patches(proj_value,
                                          sizes=[1, self.window_size, self.window_size, 1],
                                          strides=[1, 1, 1, 1],
                                          rates=[1, 1, 1, 1],
                                          padding='VALID')

        # Reshape queries, keys, and values for window
        proj_query_window = tf.reshape(proj_query, [m_batchsize, self.n_heads, C // self.n_heads, height * width])
        proj_key_window = tf.reshape(proj_key, [m_batchsize, self.n_heads, C // self.n_heads, (height + self.window_size - 1) * (width + self.window_size - 1)])
        proj_value_window = tf.reshape(window, [m_batchsize, self.n_heads, C // self.n_heads, (height + self.window_size - 1) * (width + self.window_size - 1)])

        # Calculate attention
        energy = tf.matmul(tf.transpose(proj_query_window, perm=[0, 1, 3, 2]), proj_key_window)  # (B X n_heads X HW X HW)
        attention = self.softmax(energy)

        # Apply attention to values
        out_window = tf.matmul(proj_value_window, tf.transpose(attention, perm=[0, 1, 3, 2]))  # (B X n_heads X C/n_heads X HW)

        # Reshape output window
        out_window = tf.reshape(out_window, [m_batchsize, C, height, width])
        out_window = tf.transpose(out_window, perm=[0, 2, 3, 1])

        # Residual connection
        out = self.gamma * out_window + x

        return out
    
def res_block(x, nb_filters, strides):
    initializer = tf.keras.initializers.HeNormal()
    
    res_path = keras.layers.BatchNormalization()(x)
    res_path = keras.layers.Activation(activation='relu')(res_path)
    res_path = keras.layers.Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0], kernel_initializer=initializer)(res_path)
    res_path = keras.layers.BatchNormalization()(res_path)
    res_path = keras.layers.Activation(activation='relu')(res_path)
    res_path = keras.layers.Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1], kernel_initializer=initializer)(res_path)

    shortcut = keras.layers.Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0], kernel_initializer=initializer)(x)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    res_path = keras.layers.add([shortcut, res_path])
    return res_path

def encoder(x):
    to_decoder = []
    initializer = tf.keras.initializers.HeNormal()

    main_path = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_initializer=initializer)(x)
    main_path = keras.layers.BatchNormalization()(main_path)
    main_path = keras.layers.Activation(activation='relu')(main_path)

    main_path = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_initializer=initializer)(main_path)

    shortcut = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), kernel_initializer=initializer)(x)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    main_path = keras.layers.add([shortcut, main_path])
    main_path = pcbam(main_path)
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    main_path = cbam(main_path)
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    main_path = pcbam(main_path)
    to_decoder.append(main_path)

    return to_decoder

def decoder(x, from_encoder):
    x = SWA(in_dim = 512)(x)
    
    x_1 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    x_1 = keras.layers.UpSampling2D(size=(2, 2))(x_1)
    
    x_2 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    x_2 = keras.layers.UpSampling2D(size=(2, 2))(x_2)
    x_2 = keras.layers.UpSampling2D(size=(2, 2))(x_2)
    
    x_3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    x_3 = keras.layers.UpSampling2D(size=(2, 2))(x_3)
    x_3 = keras.layers.UpSampling2D(size=(2, 2))(x_3)
    x_3 = keras.layers.UpSampling2D(size=(2, 2))(x_3)
    
    main_path = keras.layers.UpSampling2D(size=(2, 2))(x)
    main_path = keras.layers.concatenate([main_path, from_encoder[2]], axis=3)
    from_encoder[2] = from_encoder[2] + x_1
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = keras.layers.UpSampling2D(size=(2, 2))(main_path)
    main_path = keras.layers.concatenate([main_path, from_encoder[1]], axis=3)
    from_encoder[1] = from_encoder[1] + x_2
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = keras.layers.UpSampling2D(size=(2, 2))(main_path)
    main_path = keras.layers.concatenate([main_path, from_encoder[0]], axis=3)
    from_encoder[0] = from_encoder[0] + x_3
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path

def build_res_unet(input_shape):
    inputs = keras.layers.Input(shape=input_shape)
    inputs_rgb = tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)

    to_decoder = encoder(inputs_rgb)

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    return keras.models.Model(inputs=inputs, outputs=path)


model = build_res_unet(input_shape=(128, 128, 1))
optimizer = AdamW(lr=0.0001)
model.compile(loss=combined_loss, metrics=["accuracy",dice_score,recall,precision,iou], optimizer = optimizer)
model.summary()
