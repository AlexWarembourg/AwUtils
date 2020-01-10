import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Conv1D,
                                     Lambda, GRU, Dense, BatchNormalization,
                                     concatenate, Reshape, Dropout)
from tensorflow.keras.models import Model

lv1_cnt, lv2_cnt, lv3_cnt, lv4_cnt = 7, 36, 193, 975

# for reproducibility
tf.keras.backend.clear_session()
tf.set_random_seed(42)
np.random.seed(42)

if tf.test.is_gpu_available():
    print('running with gpu')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))
else:
    print('no GPU available running only with CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU

max_embed_length = 200
one_hot = False


def wavenet_struct(latent_dim, x_in):
    c1 = Conv1D(latent_dim, 2, dilation_rate=1, padding='causal', activation='relu')(x_in)
    c2 = Conv1D(latent_dim, 2, dilation_rate=2, padding='causal', activation='relu')(c1)
    c2 = Conv1D(latent_dim, 2, dilation_rate=4, padding='causal', activation='relu')(c2)
    c2 = Conv1D(latent_dim, 2, dilation_rate=8, padding='causal', activation='relu')(c2)
    c2 = Conv1D(latent_dim, 2, dilation_rate=16, padding='causal', activation='relu')(c2)
    c4 = concatenate([c1, c2])
    conv_out = Conv1D(8, 1, activation='relu')(c4)
    conv_out = Dropout(0.25)(conv_out)
    return conv_out


def smape_loss(true, predicted):
    epsilon = 0.1
    true_o = tf.math.expm1(true)
    pred_o = tf.math.expm1(predicted)
    summ = tf.maximum(tf.abs(true_o) + tf.abs(pred_o) + epsilon, 0.5 + epsilon)
    smape = tf.abs(pred_o - true_o) / summ * 2.0
    return tf.reduce_mean(smape)


def time_distributed_dense_layer(x, neurons=128, activation=None, batch_norm=None, reuse=False, dropout=0.2):
    time_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(neurons, kernel_initializer="random_normal", bias_initializer="random_normal")
    )(x)
    if batch_norm is not None:
        z = tf.layers.batch_normalization(time_layer, training=batch_norm, reuse=reuse)
    z = activation(time_layer) if activation else z
    z = tf.nn.dropout(z, dropout) if dropout is not None else z
    return z


def encoder_decoder_model(latent_dim=100, timesteps=90, range_ahead=42, stacked=False):
    # Model Input
    seq_in = Input(shape=(timesteps, 1), dtype='float32', name="demand_seq_in")
    lagged_seq = Input(shape=(timesteps, 1), dtype="float32", name="lagged_seq_in")
    promo_in = Input(shape=(timesteps + range_ahead, 1), dtype='float32', name="promo")
    weekday_in = Input(shape=(timesteps + range_ahead,), dtype='float32', name="dow")
    dom_in = Input(shape=(timesteps + range_ahead,), dtype='uint8', name="dayofmonth")
    month_in = Input(shape=(timesteps + range_ahead,), dtype='uint8', name="month")
    holiday_in = Input(shape=(timesteps + range_ahead, 1), dtype='float32', name="Holiday")
    # exposition_in = Input(shape=(timesteps + range_ahead, 1), dtype='float32', name="Exposition")

    # embedding on temporal informations
    weekday_embed_encode = Embedding(7 + 1, math.ceil(8 / 2),
                                     input_length=timesteps + range_ahead,
                                     embeddings_initializer='he_uniform',
                                     name="weekday_embedding")(weekday_in)

    dom_embed_encode = Embedding(31 + 1, math.ceil(32 / 2),
                                 input_length=timesteps + range_ahead,
                                 embeddings_initializer='he_uniform',
                                 name="dayofmonth_embedding")(dom_in)

    month_embed_encode = Embedding(12 + 1, math.ceil(13 / 2),
                                   input_length=timesteps + range_ahead,
                                   embeddings_initializer='he_uniform',
                                   name="month_embedding")(month_in)

    # aux input
    cat_features = Input(shape=(timesteps + range_ahead, 2), name="categorical_features", dtype="int64")
    sku = Lambda(lambda x: x[:, :, 0])(cat_features)
    store = Lambda(lambda x: x[:, :, 1])(cat_features)
    '''
    flg_loy = Lambda(lambda x: x[:, :, 2])(cat_features)
    lvl1_nomenc = Lambda(lambda x: x[:, :, 3])(cat_features)
    lvl2_nomenc = Lambda(lambda x: x[:, :, 4])(cat_features)
    lvl3_nomenc = Lambda(lambda x: x[:, :, 5])(cat_features)
    lvl4_nomenc = Lambda(lambda x: x[:, :, 6])(cat_features)
    format_ = Lambda(lambda x: x[:, :, 7])(cat_features)

    loyalty_ohe = Lambda(tf.keras.backend.one_hot, arguments={'num_classes': 2},
                         output_shape=(timesteps + range_ahead, 2))(flg_loy)
    format_ohe = Lambda(tf.keras.backend.one_hot, arguments={'num_classes': 2},
                        output_shape=(timesteps + range_ahead, 2))(format_)

    lv1_embed = Embedding(lv1_cnt + 1, math.ceil(lv1_cnt / 2) if lv1_cnt < max_embed_length else max_embed_length,
                          input_length=timesteps + range_ahead,
                          embeddings_initializer='he_uniform')(lvl1_nomenc)

    lv2_embed = Embedding(lv2_cnt + 1, math.ceil(lv2_cnt / 2) if lv2_cnt < max_embed_length else max_embed_length,
                          input_length=timesteps + range_ahead,
                          embeddings_initializer='he_uniform')(lvl2_nomenc)

    lv3_embed = Embedding(lv3_cnt + 1, math.ceil(lv3_cnt / 2) if lv3_cnt < max_embed_length else max_embed_length,
                          input_length=timesteps + range_ahead,
                          embeddings_initializer='he_uniform')(lvl3_nomenc)

    lv4_embed = Embedding(lv4_cnt + 1, math.ceil(lv4_cnt / 2) if lv4_cnt < max_embed_length else max_embed_length,
                          input_length=timesteps + range_ahead,
                          embeddings_initializer='he_uniform')(lvl4_nomenc)
    '''
    if one_hot:
        sku_ohe = Lambda(tf.keras.backend.one_hot, arguments={'num_classes': 33730},
                         output_shape=(timesteps + range_ahead, 33730))(sku)
        cat_ohe = Lambda(tf.keras.backend.one_hot, arguments={'num_classes': 27},
                         output_shape=(timesteps + range_ahead, 27))(store)
    else:
        sku_embed = Embedding(39805 + 1, max_embed_length,
                              input_length=timesteps + range_ahead,
                              embeddings_initializer='he_uniform')(sku)
        store_embed = Embedding(27 + 1, math.ceil(27 / 2),
                                input_length=timesteps + range_ahead,
                                embeddings_initializer='he_uniform')(store)
    # Encoder
    encode_slice = Lambda(lambda x: x[:, :timesteps, :])

    encode_features = concatenate([sku_embed,
                                   store_embed,
                                   # loyalty_ohe,
                                   # format_ohe,
                                   promo_in,
                                   # exposition_in,
                                   holiday_in,
                                   # lv1_embed,
                                   # lv2_embed,
                                   # lv3_embed,
                                   # lv4_embed,
                                   weekday_embed_encode,
                                   dom_embed_encode,
                                   month_embed_encode
                                   ], axis=2)

    encode_features = encode_slice(encode_features)
    # convolution layer at one sequential dimension (1D)
    # filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    # Kernel_size : An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    # padding : same(zero padded)|causal|valid(not zero padded)
    # conv_in = Conv1D(filters=32, kernel_size=4, padding='same', kernel_initializer='he_uniform')(seq_in)
    # conv_in = wavenet_struct(latent_dim, seq_in)
    x_encode = concatenate([seq_in, lagged_seq, encode_features], axis=2)
    # first gru layers
    encoder = GRU(latent_dim, return_state=True, recurrent_initializer='he_uniform')
    # second if stacked
    encoder2 = GRU(latent_dim, return_state=True, return_sequences=False, recurrent_initializer='he_uniform')

    print('Input dimension:', x_encode.shape)
    if not stacked:
        _, h = encoder(x_encode)
    else:
        s1, h1 = encoder(x_encode)
        s1 = Dropout(0.25)(s1)
        s1 = BatchNormalization()(s1)
        _, h = encoder2(s1)

    # Connector
    h = Dense(latent_dim, activation='tanh', kernel_initializer='he_uniform')(h)

    # Decoder
    previous_x = Lambda(lambda x: x[:, -1, :])(seq_in)

    decode_slice = Lambda(lambda x: x[:, timesteps:, :])

    # Decoder
    decode_features = concatenate([sku_embed,
                                   store_embed,
                                   # loyalty_ohe,
                                   # format_ohe,
                                   promo_in,
                                   # exposition_in,
                                   holiday_in,
                                   # lv1_embed,
                                   # lv2_embed,
                                   # lv3_embed,
                                   # lv4_embed,
                                   weekday_embed_encode,
                                   dom_embed_encode,
                                   month_embed_encode
                                   ], axis=2)

    decode_features = decode_slice(decode_features)
    decoder = GRU(latent_dim, return_state=True, return_sequences=False, recurrent_initializer='he_uniform')
    # last layer
    decoder_dense2 = Dense(1, activation='linear')
    slice_at_t = Lambda(lambda x: tf.slice(x, [0, i, 0], [-1, 1, -1]))

    for i in range(range_ahead):
        previous_x = Reshape((1, 1))(previous_x)
        features_t = slice_at_t(decode_features)
        decode_input = concatenate([previous_x, features_t], axis=2)
        output_x, h = decoder(decode_input, initial_state=h)
        output_x = decoder_dense2(output_x)
        # gather outputs
        if i == 0:
            decoder_outputs = output_x
        elif i > 0:
            decoder_outputs = concatenate([decoder_outputs, output_x])

        previous_x = output_x

    # /!\  be careful to order in format cnn return
    model = Model([seq_in, lagged_seq, holiday_in,  # exposition_in,
                   promo_in, weekday_in, dom_in,
                   month_in, cat_features], decoder_outputs)
    return model
