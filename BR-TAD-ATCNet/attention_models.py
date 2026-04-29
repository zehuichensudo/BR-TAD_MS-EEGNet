
import math
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense
from tensorflow.keras.layers import multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization, Reshape
from tensorflow.keras import backend as K


#%% 创建并应用注意力模型
def attention_block(in_layer, attention_model, ratio=8, residual = False, apply_to_input=True):
    in_sh = in_layer.shape # 输入张量的维度
    in_len = len(in_sh)
    expanded_axis = 2 # 默认值 = 2

    if attention_model == 'mha':   # 多头自注意力层
        if(in_len > 3):
            in_layer = Reshape((in_sh[1],-1))(in_layer)
        out_layer = mha_block(in_layer)
    elif attention_model == 'mhla':  # 多头局部自注意力层
        if(in_len > 3):
            in_layer = Reshape((in_sh[1],-1))(in_layer)
        out_layer = mha_block(in_layer, vanilla = False)
    elif attention_model == 'se':   # 挤压与激励（Squeeze-and-excitation）层
        if(in_len < 4):
            in_layer = tf.expand_dims(in_layer, axis=expanded_axis)
        out_layer = se_block(in_layer, ratio, residual, apply_to_input)
    elif attention_model == 'cbam': # 卷积块注意力模块（Convolutional block attention module）
        if(in_len < 4):
            in_layer = tf.expand_dims(in_layer, axis=expanded_axis)
        out_layer = cbam_block(in_layer, ratio=ratio, residual = residual)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_model))

    if (in_len == 3 and len(out_layer.shape) == 4):
        out_layer = tf.squeeze(out_layer, expanded_axis)
    elif (in_len == 4 and len(out_layer.shape) == 3):
        out_layer = Reshape((in_sh[1], in_sh[2], in_sh[3]))(out_layer)
    return out_layer


#%% 多头自注意力 (MHA) 块
def mha_block(input_feature, key_dim=8, num_heads=2, dropout = 0.5, vanilla = True):
    """多头自注意力 (MHA) 块。

    这里我们包含了两种类型的 MHA 块：
            原始的多头自注意力，如 https://arxiv.org/abs/1706.03762 所述
            多头局部自注意力，如 https://arxiv.org/abs/2112.13492v1 所述
    """
    # 层归一化
    x = LayerNormalization(epsilon=1e-6)(input_feature)

    if vanilla:
        # 创建一个多头注意力层，如
        # 'Attention Is All You Need' https://arxiv.org/abs/1706.03762 所述
        x = MultiHeadAttention(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(x, x)
    else:
        # 创建一个多头局部自注意力层，如
        # 'Vision Transformer for Small-Size Datasets' https://arxiv.org/abs/2112.13492v1 所述

        # 构建对角注意力掩码
        NUM_PATCHES = input_feature.shape[1]
        diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

        # 创建一个多头局部自注意力层。
        # x = MultiHeadAttention_LSA(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(
        #     x, x, attention_mask = diag_attn_mask)
        x = MultiHeadAttention_LSA(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(
            x, x, attention_mask = diag_attn_mask)
    x = Dropout(0.3)(x)
    # 残差连接 (Skip connection)
    mha_feature = Add()([input_feature, x])

    return mha_feature


#%% 多头自注意力 (MHA) 块：局部自注意力 (LSA)
class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
    """局部多头自注意力块

     局部自注意力如 https://arxiv.org/abs/2112.13492v1 所述
     此实现取自 https://keras.io/examples/vision/vit_small_ds/
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 可训练的温度项。初始值是键维度（key dimension）的平方根。
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


#%% 挤压与激励 (Squeeze-and-excitation) 块
def se_block(input_feature, ratio=8, residual = False, apply_to_input=True):
    """(Squeeze-and-Excitation, SE) 块。

    如 https://arxiv.org/abs/1709.01507 所述
    该实现取自 https://github.com/kobiso/CBAM-keras
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if (ratio != 0):
        se_feature = Dense(channel // ratio,
                           activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')(se_feature)
        assert se_feature.shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    if(apply_to_input):
        se_feature = multiply([input_feature, se_feature])

    # 残差连接
    if(residual):
        se_feature = Add()([se_feature, input_feature])

    return se_feature


#%% 卷积块注意力模块 (Convolutional block attention module)
def cbam_block(input_feature, ratio=8, residual = False):
    """ 卷积块注意力模块 (CBAM) 块。

    如 https://arxiv.org/abs/1807.06521 所述
    该实现取自 https://github.com/kobiso/CBAM-keras
    """

    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)

    # 残差连接
    if(residual):
        cbam_feature = Add()([input_feature, cbam_feature])

    return cbam_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
#     channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])