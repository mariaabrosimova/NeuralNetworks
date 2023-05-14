import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
pathToData = 'C:\\Users\\Мария\\PycharmProjects\\Lab6\\lab6\\'
fn_model1 = pathToData +'m1'+ 'lk3.h5'
fn_model2 = pathToData +'m2' +'lk3.h5'
pathToHistory = 'C:\\Users\\Мария\\PycharmProjects\\Lab6\\'
suff = '.txt'

# Имена файлов, в которые сохраняется история обучения
fn_loss = pathToHistory + 'loss_' + suff
fn_acc = pathToHistory + 'acc_' + suff
fn_val_loss = pathToHistory + 'val_loss_' + suff
fn_val_acc = pathToHistory + 'val_acc_' + suff

def WriteHistorytoTextFile(history_buf):
    with open(fn_loss, 'w') as output:
        for val in history_buf['loss']:output.write(str(val) + '\n')
    with open(fn_acc, 'w') as output:
        for val in history_buf['accuracy']: output.write(str(val) + '\n')
    with open(fn_val_loss, 'w') as output:
        for val in history_buf['val_loss']: output.write(str(val) + '\n')
    with open(fn_val_acc, 'w') as output:
        for val in history_buf['val_accuracy']: output.write(str(val) + '\n')

epochs = 10
(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
x_trn = x_trn.reshape(-1, 28, 28, 1)




generation = ImageDataGenerator(featurewise_center= True)
generation.fit(x_trn)
x_y = generation.flow(x_trn, y_trn, batch_size = len(y_trn), shuffle=False)[0][0][:].astype('uint8')

#Кодер и декодер автокодировщика построены из блоков,
#формируемых процедурой one_part:
def one_part(units, x):
    x = Dense(units)(x)
    x = LeakyReLU()(x)
    return Dropout(0.3)(x)
def some_plts(imgs):
    fig, axs = plt.subplots(5, 5)
    k = -1
    for i in range(5):
        for j in range(5):
            k += 1
            img = imgs[k].reshape(28, 28)
            axs[i, j].imshow(img, cmap = 'gray')
            axs[i, j].axis('off')
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.show()
some_plts(x_trn)
some_plts(x_y)

#Формирование модели автокодировщика

latent_size = 32 # Размер латентного пространста
inp = Input(shape = (784))
x = one_part(512, inp)
x = one_part(256, x)
x = one_part(128, x)
x = one_part(64, x)
x = Dense(latent_size)(x)
encoded = LeakyReLU()(x)
x = one_part(64, encoded)
x = one_part(128, x)
x = one_part(256, x)
x = one_part(512, x)
decoded = Dense(784, activation = 'sigmoid')(x)
model = Model(inputs = inp, outputs = decoded)
model.compile('adam', loss = 'binary_crossentropy') # nadam
model.summary()

x_y = x_y.reshape(-1, 784) / 255.
x_trn = x_trn.reshape(-1, 784) / 255.



history = model.fit(x_trn, x_y, epochs = epochs, batch_size = 256)
x_tst = x_tst.reshape(x_tst.shape[0], 28, 28, 1)
generation_test = ImageDataGenerator(featurewise_center=True)
generation_test.fit(x_tst)

#history_buf1 = history.history
#WriteHistorytoTextFile(history_buf1)


x_tst1 = generation.flow(x_tst, y_tst, batch_size=len(y_tst), shuffle=False)[0][0][:].astype('uint8')
x_tst = x_tst.reshape(-1, 784) / 255.0
some_plts(x_tst)
some_plts(x_tst1)

#Прогноз автокодировщика
predicted_images = model.predict(x_tst)
some_plts(predicted_images)

# опредедлить точность генерации

i = np.random.randint(len(y_trn))
img = x_trn[i].reshape(28, 28)
def one_plt(img):
    plt.figure(figsize = (2, 2))
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.show()
one_plt(img)



for i in range(10):
    img1 = x_trn[i].reshape(28, 28, 1)
    img2 = x_y[i].reshape(28, 28, 1)
    ssim = tf.image.ssim(img1, img2, 1)
    print(ssim)

predicted_images = model.predict(x_tst)

for i in range(25):
    plt.subplot(5, 5, i + 1)
    img = predicted_images[i].reshape(28, 28, 1)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    x = np.round(
        ssim(x_trn[i].reshape(28, 28, 1), predicted_images[i].reshape(28, 28, 1), multichannel=True),
        4)
    plt.title(x)
    plt.axis('off')

plt.subplots_adjust(hspace=0.5)
plt.show()



#img1 = x_trn[0].reshape(28, 28,1)
#img2 = x_y[0].reshape(28, 28,1)
#ssim = skimage.measure.compare_ssim(ref_array, img_array, multichannel=True, data_range=255)
#im1 = tf.image.convert_image_dtype(im1, tf.float32)
#im2 = tf.image.convert_image_dtype(im2, tf.float32)

#ssim = tf.image.ssim(img1, img2, 1)#Тензор, содержащий значение SSIM для каждого изображения в пакете.
# Возвращаемые значения SSIM находятся в диапазоне (-1, 1], когда значения пикселей неотрицательны
#print(ssim)

