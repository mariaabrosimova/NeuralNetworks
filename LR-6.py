import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten,Reshape
from mnist import MNIST
from keras.models import Model
import keras.backend as K
import time
from keras.models import load_model
#--------------------const--------------------------
flagmnist = True
reshape_vice_flatten = True
fldroput = True
switch_off = False
pathToData = 'C:\\Users\\Мария\\PycharmProjects\\Lab6\\lab6\\'
fn_model1 = pathToData +'m1'+ 'lk3.h5'
fn_model2 = pathToData +'m2' +'lk3.h5'
pathToHistory = 'C:\\Users\\Мария\\PycharmProjects\\Lab6\\'
suff = '.txt'
img_rows = img_cols = 28
num_classes = 0
numberepochs = 10

# Имена файлов, в которые сохраняется история обучения
fn_loss = pathToHistory + 'loss_' + suff
fn_acc = pathToHistory + 'acc_' + suff
fn_val_loss = pathToHistory + 'val_loss_' + suff
fn_val_acc = pathToHistory + 'val_acc_' + suff
map_for_emnist = dict(zip(np.arange(0, 26),['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                                            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                            'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                                            'y', 'z']))
#--------------------functions--------------------------
def one_plot(n, y_lb, loss_acc, val_loss_acc):
    plt.subplot(1, 2, n)
    if n == 1:
        lb, lb2 = 'loss', 'val_loss'
        yMin = 0
        yMax = 1.05 * max(max(loss_acc), max(val_loss_acc))
    else:
        lb, lb2 = 'acc', 'val_acc'
        yMin = min(min(loss_acc), min(val_loss_acc))
        yMax = 1.0
    plt.plot(loss_acc, color = 'r', label = lb, linestyle = '--')
    plt.plot(val_loss_acc, color = 'g', label = lb2)
    plt.ylabel(y_lb)
    plt.xlabel('Эпоха')
    plt.ylim([0.95 * yMin, yMax])
    plt.legend()
def OutputData25(flagmnist, x_train):
    x_buf = x_train.copy()
    y_buf = y_train.copy()
    if flagmnist:
        name = "MNIST"
    else:
        name ="EMNIST"
    plt.figure(figsize=(10, 10))
    plt.suptitle(name)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        if (flagmnist):
            plt.imshow(x_buf.reshape(-1, 28, 28, 1)[i], cmap=plt.cm.binary)
            plt.title(y_buf[i], size=7, weight="heavy")
        else:
            plt.imshow(x_buf.reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)[i], cmap=plt.cm.binary)
            plt.title(map_for_emnist[y_buf[i]], size=7, weight="heavy")
    plt.show()
#запись данных в бинарные файлы
def DatatoBIN(n):
    buf_path= "C:\\Users\\Мария\\PycharmProjects\\Lab6\\lab6"
    buf_path = buf_path + '\\' + n
    mndata = MNIST(buf_path)
    mndata.gz = True
    imagesTrain, labelsTrain = mndata.load_training()
    imagesTest, labelsTest = mndata.load_testing()
    f1 = open(buf_path +'imagesTrain.bin', 'wb')
    f2 = open(buf_path +'labelsTrain.bin', 'wb')
    f3 = open(buf_path +'imagesTest.bin', 'wb')
    f4 = open(buf_path + 'labelsTest.bin', 'wb')
    f1.write(np.uint8(imagesTrain))
    f2.write(np.uint8(labelsTrain))
    f3.write(np.uint8(imagesTest))
    f4.write(np.uint8(labelsTest))
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    print(n+'-данные сохранены в двоичные файлы')
#запись данных из двоичных файлов в массивы
def ReadBIN(n):
    buf_path = "C:\\Users\\Мария\\PycharmProjects\\Lab6\\lab6"
    buf_path = buf_path + '\\' + n
    print('Загрузка данных из двоичных файлов')
    with open(buf_path + 'imagesTrain.bin', 'rb') as read_binary:
        img1 = np.fromfile(read_binary, dtype = np.uint8)
    with open(buf_path + 'labelsTrain.bin', 'rb') as read_binary:
        label1 = np.fromfile(read_binary, dtype = np.uint8)
    with open(buf_path + 'imagesTest.bin', 'rb') as read_binary:
        img2 = np.fromfile(read_binary, dtype = np.uint8)
    with open(buf_path + 'labelsTest.bin', 'rb') as read_binary:
        label2 = np.fromfile(read_binary, dtype = np.uint8)
    return img1, label1, img2, label2
def WriteHistorytoTextFile(history_buf):
    with open(fn_loss, 'w') as output:
        for val in history_buf['loss']:output.write(str(val) + '\n')
    with open(fn_acc, 'w') as output:
        for val in history_buf['accuracy']: output.write(str(val) + '\n')
    with open(fn_val_loss, 'w') as output:
        for val in history_buf['val_loss']: output.write(str(val) + '\n')
    with open(fn_val_acc, 'w') as output:
        for val in history_buf['val_accuracy']: output.write(str(val) + '\n')
def BuildPlot(history):
    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(wspace=0.5)
    one_plot(1, 'Потери', history['loss'], history['val_loss'])
    one_plot(2, 'Точность', history['accuracy'], history['val_accuracy'])
    plt.suptitle('Потери и точность')
    plt.show()
def buf_x_y(num_classes):
    if (num_classes == 26):
        buf = 1
    else:
        buf = 0
    x_train = np.asarray(imagesTrain)
    y_train = np.asarray(labelsTrain)-buf
    x_test = np.asarray(imagesTest)
    y_test = np.asarray(labelsTest)-buf
    x_train = np.array(x_train, dtype='float32') / 255 #стандартизация
    x_test = np.array(x_test, dtype='float32') / 255
    # преобразование в вектор, размерность которого равна кол-ву классов задачи
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test, y_train_cat, y_test_cat

def predictEMNIST(n_test, predicted_classes, true_classes, m, lst_false, m_max, false_classified, model, number_model):
    print('Индекс  | Прогноз код  |  Правильный класс, код    | Прогноз (буква)   | Правильная буква')
    i=0
    while i < n_test:
        cls_pred = predicted_classes[i]  # Предсказанное моделью имя класса
        cls_true = true_classes[i]  # Истинное имя класса
        if cls_pred != cls_true:
            m += 1
            lst_false.append([i, cls_pred, cls_true])
            if (m == min(m_max, false_classified)): break
            print("%-i%20i%20i%20s%20s" % (i, cls_pred, cls_true, map_for_emnist[cls_pred], map_for_emnist[ cls_true]))
            i += 800
        else:
            i += 1
    plt.figure("Ошибки классификации. Модель %d" % (number_model))
    for k in range(len(lst_false)):
        plt.subplot(3, 5, k + 1)
        lst = lst_false[k]
        plt.imshow(x_test[lst[0]].reshape(img_rows, img_cols), cmap='gray')
        plt.title('{}/{}'.format(map_for_emnist[lst[1]], map_for_emnist[lst[2]]))
        plt.axis('off')
    plt.show()
    print("Точность по классам")
    for i in range(num_classes):
        predictions = model.predict(x_test[y_test == i])
        n = sum(np.round(predictions[ : ,i]) == 1)
        print(map_for_emnist[i], ": ", n / sum(y_test == i))
def predictMNIST(n_test, predicted_classes, true_classes, m, lst_false, m_max, false_classified, model, number_model):
    print('Индекс  | Прогноз  |  Правильный класс')
    for i in range(n_test):  #
        cls_pred = predicted_classes[i]  # Предсказанное моделью имя класса
        cls_true = true_classes[i]  # Истинное имя класса
        if cls_pred != cls_true:
            m += 1
            lst_false.append([i, cls_pred, cls_true])
            if (m == min(m_max, false_classified)): break
            print("%-i%10i%10i" % (i, cls_pred, cls_true))
    #plt.figure('Ошибки классификации в модели {:.i%} ' .format(number_model))
    plt.figure("Ошибки классификации. Модель %d" % (number_model))

    for k in range(len(lst_false)):
        plt.subplot(3, 5, k + 1)
        lst = lst_false[k]
        plt.imshow(x_test[lst[0]].reshape(img_rows, img_cols), cmap='gray')
        plt.title('{}/{}'.format(lst[1], lst[2]))
        plt.axis('off')
    plt.show()
    print("Точность по классам")
    for i in range(num_classes):
        predictions = model.predict(x_test[y_test == i])
        n = sum(np.round(predictions[:, i]) == 1)
        print(i, ": ", n / sum(y_test == i))

def Forecast(fn_model_x):
    if (fn_model_x == fn_model1):
        number_model = 1
    else:
        number_model = 2
    model = load_model(fn_model_x)#(fn_model2)
    score = model.evaluate(x_test, y_test_cat, verbose = 0)
    # Вывод потерь и точности
    print('Потери при тестировании:', round(score[0], 4))
    print('Точность при тестировании: {}{}'.format(score[1] * 100, '%'))
    # Прогноз
    y_pred = model.predict(x_test) #метки классов, предсказанных моделью НС
    predicted_classes = np.array([np.argmax(m) for m in y_pred])
    true_classes = np.array([np.argmax(m) for m in y_test_cat])
    n_test = len(y_test)
    print("Всего изображений в тестовой выборке: ", n_test)
    true_classified = np.sum(predicted_classes == true_classes)
    print("Число верно классифицированных изображений: ", true_classified)
    false_classified = n_test - true_classified
    acc = 100.0 * true_classified / n_test
    print('Точность: {}{}'.format(acc, '%'))
    print('Неверно классифицированно:', false_classified)
    m, m_max = 0, 15
    lst_false = []
    i = 0
    if (flagmnist):
        predictMNIST(n_test, predicted_classes, true_classes, m, lst_false, m_max, false_classified, model, number_model)
    else:
        predictEMNIST(n_test, predicted_classes, true_classes, m, lst_false, m_max, false_classified, model, number_model)
#--------------------------------------------------------main--------------------------------------------------------------------------------------
#флаги
#MNIST/EMNIST
#flagmnist = True
flagmnist = False
#использование reshape вместо flatten
reshape_vice_flatten = True
#reshape_vice_flatten = False #
#включение слоя dropout
fldroput = True
#fldroput = False
#switch_off = True
switch_off = False


if (flagmnist):
    n= 'mnist'
    num_classes = 10
    x_train_buf_number = 60000
    x_test_buf_number = 10000
else:
    n= 'emnist_letters'
    num_classes = 26
    x_train_buf_number = 124800 #обучающая выборка
    x_test_buf_number = 20800 #выборка валидации

print("______________________________________________________________")
print("dataset: ", n)
print("number of classes: ", num_classes)
DatatoBIN(n)
imagesTrain, labelsTrain, imagesTest, labelsTest = ReadBIN(n)
#1 модель
x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = buf_x_y(num_classes)
OutputData25(flagmnist, x_train)
print("")
if switch_off:
    print("отключение слоев reshape и flatten")
    size = img_rows * img_cols
    x_train = x_train.reshape(-1, size)
    x_test = x_test.reshape(-1, size)
    input_shape = (size)  # 784
else:
    if reshape_vice_flatten:
        print("reshape вместо слоя flatten")
        size = img_rows * img_cols
        x_train = x_train.reshape(-1, size)
        x_test = x_test.reshape(-1, size)
        input_shape = (size)  # 784
    else:
        print("flatten вместо слоя reshape")
        x_train = x_train.reshape(-1, img_rows, img_cols)
        x_test = x_test.reshape(-1, img_rows, img_cols)
        input_shape = (img_rows, img_cols)
print("")

#model 1
K.clear_session()
model1 = keras.models.Sequential()
model1.add(keras.layers.Dense(128, input_dim=784, activation='elu'))
model1.add(keras.layers.Dense(64, input_dim=784, activation='elu'))
model1.add(keras.layers.Dense(32, activation='elu'))
model1.add(keras.layers.Dense(num_classes, activation='softmax'))
print("------------------------------------------------------------")
print("многослойный перцептрон модель 1")
print("------------------------------------------------------------")
print(model1.summary())
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#input_shape = (size)  # 784
#categorical_crossentropy- минимизирует кросс-энтропию, metrics=['accuracy'] выводит процент правильных ответов
#x_train.reshape(-1, img_rows * img_cols)
start = time.time()
history = model1.fit(
 x_train.reshape(-1, img_rows * img_cols),
 y_train_cat,
 batch_size=64,#  64 объекта для подсчета градиента на каждом шаге, default = 32
 epochs=numberepochs,# numberepochs проходов по датасету
 validation_data=(x_train.reshape(-1, img_rows * img_cols), y_train_cat))
end = time.time()
print(end - start)
history_buf1 = history.history
WriteHistorytoTextFile(history_buf1)
BuildPlot(history_buf1)
model1.save(fn_model1)

#model 2
print("------------------------------------------------------------")
print("многослойный перцептрон модель 2") #пример из документа
print("------------------------------------------------------------")
inp = Input(shape = input_shape)# Входной слой
x = inp
if (not(switch_off)):
    if reshape_vice_flatten:
        x = Reshape(target_shape=(-1,))(x)

    else:
        x = Flatten()(x)# Преобразование 2D в 1D

x = Dense(784, activation = 'relu')(x) #relu #72 was c 72 было точнее #784   32
if fldroput:
    x = keras.layers.Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output = Dense(num_classes, activation = 'softmax')(x)
model2 = Model(inputs=inp, outputs = output)
model2.summary()

#model2.compile(optimizer = 'Adam',loss = 'mse' , metrics = ['accuracy'])# модель 2 отчета
model2.compile(optimizer = 'Adam',loss = 'categorical_crossentropy' , metrics = ['accuracy']) #модель 3 отчета
# Обучение нейронной сети
start = time.time()
history2 = model2.fit(x_train, y_train_cat,batch_size = 64,  epochs = numberepochs, verbose = 1, validation_data = (x_test, y_test_cat)) #batch_size = 32,
end = time.time()
print("время обучения: ", end - start)
history_buf2 = history2.history
WriteHistorytoTextFile(history_buf2)
BuildPlot(history_buf2)
model2.save(fn_model2)
#-----------------------прогноз-----------------------------
print("Прогноз ")
Forecast(fn_model2)
#Forecast(fn_model1)
exit()










