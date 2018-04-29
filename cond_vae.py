# coding: utf-8
from keras.datasets import mnist
from keras import Model
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import CSVLogger
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import generate_numvec
from vae_loss import vae_loss

class Cond_VAE(object):
    def __init__(self, x_shape, y_shape, latent_dim, intermediate_dim_en, intermediate_dim_de, epsilon_std, batch_size):
        self.input_shape = x_shape
        self.category_shape = y_shape
        self.latent_dim = latent_dim
        self.intermediate_dim_en = intermediate_dim_en
        self.intermediate_dim_de = intermediate_dim_de
        self.epsilon_std = epsilon_std
        self.batch_size = batch_size
    
    def vae_model(self):
        # encoder: q(z|x, c)
        x = Input(shape=(self.input_shape[1],))
        cond = Input(shape=(self.category_shape[1],))
        inputs = concatenate([x, cond])
        hidden = Dense(self.intermediate_dim_en, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, activation='linear')(hidden)
        z_sigma = Dense(self.latent_dim, activation='linear')(hidden)

        # decoder: p(x|z, c)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_sigma])
        z_cond = concatenate([z, cond])
        dense = Dense(self.intermediate_dim_de, activation='relu')(z_cond)
        x_decoded_mean = Dense(self.input_shape[1], activation='sigmoid')(dense)
        y = vae_loss()([x, x_decoded_mean, z_mean, z_sigma])
        return Model([x, cond], y), Model([x, cond], z_mean)

    # サンプル生成用デコーダ
    def generator(self, _model):
        _, _, dense, x_decoded_mean, _ = _model.layers[6:]
        
        decoder_input = Input(shape=(self.latent_dim + self.category_shape[1],))
        _dense = dense(decoder_input)
        _x_decoded_mean = x_decoded_mean(_dense)

        return Model(decoder_input, _x_decoded_mean)

    def sampling(self, args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
        return z_mean + K.exp(z_sigma / 2) * epsilon

    def model_compile(self, model):
        model.compile(optimizer=Adam(lr=0.001), loss=None)
    
def main():
    # 定数・ハイパーパラメータ
    batch_size = 250
    latent_dim = 2
    intermediate_dim_en = 512
    intermediate_dim_de = 512
    epochs = 10
    epsilon_std = 1.0

    # データセットを呼び出し
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # ピクセル数を計算
    nb_pixel = np.prod(x_train.shape[1:])
    # 正規化する
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # concat用にベクトルの変換をする
    x_train = x_train.reshape((len(x_train), nb_pixel))
    x_test = x_test.reshape((len(x_test), nb_pixel))
    # one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # check
    print('x_test shape: {0}'.format(x_test.shape))
    print('x_train shape: {0}'.format(x_train.shape))
    print('y_test shape: {0}'.format(y_test.shape))
    print('y_train shape: {0}'.format(y_train.shape))

    # VAEクラスからインスタンスを生成
    _vae = Cond_VAE(x_train.shape, y_train.shape, latent_dim, intermediate_dim_en, intermediate_dim_de, epsilon_std, batch_size)

    # save history to CSV
    callbacks = []
    callbacks.append(CSVLogger("history.csv"))

    # build -> compile -> summary -> fit
    _model, _encoder = _vae.vae_model()
    _vae.model_compile(_model)
    _model.summary()

    _hist = _model.fit([x_train, y_train],
        verbose=1,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test], None),
        callbacks=callbacks)
    
    # save weights
    fpath = 'conv_vae_mnist_weights_' + str(epochs) + '.h5'
    _model.save_weights(fpath)

    # plot loss
    loss = _hist.history['loss']
    val_loss = _hist.history['val_loss']
    plt.plot(range(1, epochs), loss[1:], marker='.', label='loss')
    plt.plot(range(1, epochs), val_loss[1:], marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    # 散布図を描画するメソッド: scatter(データx, y, 色c)
    # show q(z|x, y) ~ p(z|y)
    x_test_encoded = _encoder.predict([x_test, y_test], batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c = np.argmax(y_test, axis=1))
    plt.colorbar()
    plt.show()

    # 1つプロットしてみる
    sample = generate_numvec(3)
    print(sample)
    _generator = _vae.generator(_model)
    plt.figure(figsize=(3, 3))
    plt.imshow(_generator.predict(sample).reshape(28,28), cmap='gray')
    plt.show()

    # いっぱいプロットしてみる p(x|z, c)
    dig = 4
    sides = 8
    max_z = 1.5

    img_it = 0
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        for j in range(0, sides):
            z2 = (((j / (sides-1)) * max_z)*2) - max_z
            z_ = [z1, z2]
            print(z_)
            vec = generate_numvec(dig, z_)
            decoded = _generator.predict(vec)
            
            plt.subplot(sides, sides, 1 + img_it)
            img_it +=1
            plt.imshow(decoded.reshape(28, 28), cmap='gray')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
    plt.show()

if __name__ == '__main__':
    main()
