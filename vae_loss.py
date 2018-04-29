# coding: utf-8
from keras.layers import Layer
from keras import backend as K
from keras import metrics

# loss function layer
class vae_loss(Layer):
  def __init__(self, **kwargs):
    self.is_placeholder = True
    super(vae_loss, self).__init__(**kwargs)

  def vae_loss(self, x, x_decoded_mean, z_sigma, z_mean):
    # E[log P(X|z, y)]
    # 正確には予測できないので，適当なサンプリングを用いて，代用する．
    reconst_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1) 
    # 事前分布と事後分布のD_KLの値
    kl_loss = 0.5 * K.sum(-1. + K.exp(z_sigma) + K.square(z_mean) - z_sigma, axis=-1)
    return K.mean(reconst_loss + kl_loss)

  def call(self, inputs):
    x = inputs[0]
    x_decoded_mean = inputs[1]
    z_sigma = inputs[2]
    z_mean = inputs[3]
    loss = self.vae_loss(x, x_decoded_mean, z_sigma, z_mean)
    self.add_loss(loss, inputs=inputs)
    return x