from sys import exit
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from mnist import MNIST
import os
import random
import sklearn
from sklearn import metrics
import time
pathToData = 'C:\\Users\\Мария\\PycharmProjects\\Lab6\\lab6\\'
e_number_train = 124800
e_number_test = 20800
map_for_emnist = dict(zip(np.arange(0, 26),['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                                            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                            'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                                            'y', 'z']))

trn_loss_out = []
tst_loss_out = []
trn_acc_out = []
tst_acc_out = []

wrong_index = []
wrong_size = []
#запись данных в бинарные файлы
def DatatoBIN(n):
    #buf_path= "C:\\Users\\Мария\\PycharmProjects\\Lab6\\lab6"
    buf_path='C:\\Users\\Мария\\PycharmProjects\\resources\\'
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
    #buf_path = "C:\\Users\\Мария\\PycharmProjects\\Lab6\\lab6"
    buf_path='C:\\Users\\Мария\\PycharmProjects\\resources\\'
    buf_path = buf_path + '\\' + n
    print('Загрузка данных из двоичных файлов для ' + n)
    with open(buf_path + 'imagesTrain.bin', 'rb') as read_binary:
        img1 = np.fromfile(read_binary, dtype = np.uint8)
    with open(buf_path + 'labelsTrain.bin', 'rb') as read_binary:
        label1 = np.fromfile(read_binary, dtype = np.uint8)
    with open(buf_path + 'imagesTest.bin', 'rb') as read_binary:
        img2 = np.fromfile(read_binary, dtype = np.uint8)
    with open(buf_path + 'labelsTest.bin', 'rb') as read_binary:
        label2 = np.fromfile(read_binary, dtype = np.uint8)
    return img1, label1, img2, label2

mnist_number_class_0 = 10
def OutputData25(x, y):
    x_buf = x.copy()
    y_buf = y.copy()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        random_index = random.randint(0, x.shape[0] - mnist_number_class_0)
        if (y[random_index] == 10):
            plt.title("letter", size=7, weight="heavy")
            plt.imshow(x_buf.reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)[random_index], cmap='binary')
            #plt.imshow(x_buf.reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)[random_index], cmap='gray')
        else:
            plt.imshow(x_buf.reshape(-1, 28, 28, 1)[random_index], cmap='gray')
            plt.title(y_buf[random_index], size=7, weight="heavy")

    plt.subplots_adjust(hspace=0.7)
    plt.show()

def buf_x_y(num_classes, imagesTrain, labelsTrain, imagesTest, labelsTest):
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
    return x_train, y_train, x_test, y_test
def OutputData25_emnist(x_train, e_y_train):
    x_buf = x_train.copy()
    y_buf = y_train.copy()
    name = "EMNIST"
    plt.figure(figsize=(10, 10))
    plt.suptitle(name)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_buf.reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)[i], cmap=plt.cm.binary)
        plt.title(map_for_emnist[y_buf[i]], size=12, weight="heavy")
    plt.subplots_adjust(hspace=0.7)
    plt.show()


def out_right(output, x_test, y_test):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        bias = random.randint(0, x_test.shape[0] - 10)
        plt.xticks([])
        plt.yticks([])
        if output[bias] != 10:
            index = '{} / {}'.format(output[bias], y_test[bias])
        else:
            if y_test[bias] != 10:
                index = 'letter / {}'.format(y_test[bias])
            else:
                index = 'letter / letter'

        plt.title(index)

        plt.imshow(x_test[bias], cmap=plt.get_cmap('gray'))
    plt.subplots_adjust(hspace=0.7)
    plt.show()

def Plot_loss_accuracy(trn_loss_out,tst_loss_out, trn_acc_out, tst_acc_out):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    #plt.suptitle('Train data')
    ax[0].plot(np.arange(len(trn_loss_out)), trn_loss_out, label='train loss', color = 'r')
    ax[0].plot(np.arange(len(trn_loss_out)), tst_loss_out, label='test loss', color='g')
    ax[1].plot(np.arange(len(tst_acc_out)), tst_acc_out, label='test acc', color='g')
    ax[1].plot(np.arange(len(trn_acc_out)), trn_acc_out, label='train accuracy', color = 'r')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # fig, ax = plt.subplots(1, 2,figsize=(20, 5))
    # plt.suptitle('Test data')
    # ax[0].plot(np.arange(len(trn_loss_out)), tst_loss_out, label='test loss', color = 'r')
    # ax[1].plot(np.arange(len(tst_acc_out)), tst_acc_out, label='test acc', color = 'g')
    # ax[0].legend()
    # ax[1].legend()
    # plt.show()
#------------------------------------------------main-------------------------------------------------------------------
#DatatoBIN('mnist')
#DatatoBIN('emnist_letters')

imagesTrain, labelsTrain, imagesTest, labelsTest =ReadBIN('mnist')
x_train, y_train, x_test, y_test = buf_x_y(10, imagesTrain, labelsTrain, imagesTest, labelsTest)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# #OutputData25(x_train, y_train)
print("")
e_imagesTrain, e_labelsTrain, e_imagesTest, e_labelsTest = ReadBIN('emnist_letters')
e_x_train, e_y_train, e_x_test, e_y_test = buf_x_y(26, e_imagesTrain, e_labelsTrain, e_imagesTest, e_labelsTest)
e_x_train = e_x_train.reshape(-1, 28, 28, 1)
e_x_test = e_x_test.reshape(-1, 28, 28, 1)
#OutputData25_emnist(e_x_train, e_y_train)

def add_20_class_with_231_elem(e_x_train, e_y_train, e_number_train):
    x = np.zeros(4620 * 28 * 28 * 1)
    x = x.reshape(4620, 28, 28, 1)
    for j in range(20):
        k = 0
        class_i = j
        fl = True
        i = 0
        while fl and (i < e_number_train):
            if (k == 231):
                fl = False
            else :
                if ((e_y_train[i]) == class_i):
                    x[k + 231 * j, ...] = e_x_train[i, ...]
                    k += 1
                i += 1
    return x

def add_6_class_with_230_elem(e_x_train, e_y_train, e_number_train):
    x = np.zeros(1380 * 28 * 28 * 1)
    x = x.reshape(1380, 28, 28, 1)
    for j in range(6):
        k = 0
        class_i =20 + j
        fl = True
        i = 20
        while fl and (i < e_number_train):
            if (k == 230):
                fl = False
            else :
                if ((e_y_train[i]) == class_i):
                    x[k + 230 * j, ...] = e_x_train[i, ...]
                    k += 1
                i += 1
    return x
def add_test(e_x_test, e_y_test, e_number_test):
    x = np.zeros(1300 * 28 * 28 * 1)
    x = x.reshape(1300, 28, 28, 1)
    for j in range(26):
        k = 0
        class_i = j
        fl = True
        i = 0
        while fl and (i < e_number_test):
            if (k == 50):
                fl = False
            else :
                if ((e_y_test[i]) == class_i):
                    x[k + 50*j, ...] = e_x_test[i, ...]
                    k += 1
                i += 1
    return x
x_20 = add_20_class_with_231_elem(e_x_train, e_y_train, e_number_train)
x_6 = add_6_class_with_230_elem(e_x_train, e_y_train, e_number_train)
x_11_class = np.vstack((x_20, x_6))
# plt.imshow(x_11_class.reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)[4849], cmap='gray')
# plt.show()
x_train = np.vstack((x_train, x_11_class))

#класс emnist-letters будет 10 классом, чтобы различаться от 0-9 классов цифр
y_11_class = np.ones(6000, dtype=np.int64)
y_11_class *= 10
y_train = np.concatenate((y_train, y_11_class))
#OutputData25(x_train, y_train)

#для тестового множества
x_11_test = add_test(e_x_test, e_y_test, e_number_test)
x_test = np.vstack((x_test, x_11_test))
# plt.imshow(x_11_test.reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)[1200], cmap='gray')
# plt.show()
y_11_test = np.ones(1300, dtype=np.int64)
y_11_test *= 10
y_test = np.concatenate((y_test, y_11_test))
OutputData25(x_test, y_test)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#создаем обучающие и проверочные пакеты из пар изображение-метка
trn_data = DataLoader(
    [[img, label] for img, label in zip(x_train.reshape(-1, 1, 28, 28).astype(np.float32), y_train)],
    batch_size=320, shuffle=True)
tst_data = DataLoader(
    [[img, label] for img, label in zip(x_test.reshape(-1, 1, 28, 28).astype(np.float32), y_test)],
    batch_size=320, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.max_pool2d1 = nn.MaxPool2d(2)
        self.max_pool2d2 = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(in_features=1024, out_features=16, bias=True)
        self.fc2 = nn.Linear(in_features=16, out_features=11, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool2d1(x)
        x = self.conv2_drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2d2(x)  # torch.Size([256, 32, 5, 5])
        #x = torch.flatten(x, 1)
        #x = x.view(-1, 800)  # torch.Size([256, 800]) (32 * 5 * 5 = 800)
        x = x.view(-1, 1024)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)



model = Net()
print(model)
criterion = nn.CrossEntropyLoss()# Функция потерь перекрестная энтропия

def train(epoch):
    start_time = time.time()
    trn_loss = tst_loss = 0
    trn_acc = tst_acc = 0
    model.train()# Режим обучения
    for i, (data, target) in enumerate(trn_data):
        optimizer.zero_grad()#Устанавливает градиенты всех оптимизированных torch.Tensor на ноль.
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        trn_loss += loss.item() * data.size(0)
        trn_acc = np.sum(np.array([np.argmax(item) for item in output.detach().numpy()]) == target.detach().numpy()) / \
                  output.detach().numpy().shape[0]

    model.eval()
    for data, target in tst_data:
        output = model(data)
        #cредние потери
        loss = criterion(output, target)
        tst_loss += loss.item() * data.size(0)
        tst_acc = np.sum(np.array([np.argmax(item) for item in output.detach().numpy()]) == target.detach().numpy()) / \
                  output.detach().numpy().shape[0]

    trn_loss = trn_loss / x_train.shape[0]
    tst_loss = tst_loss / x_test.shape[0]
    if len(trn_loss_out) >= 2:
        if tst_loss_out[len(trn_loss_out) - 1] > tst_loss:
            torch.save(model.state_dict(), 'model_weights.pth')  #сохранение весов модели

    print('--------------------------------------------------------------------------------------')
    print('Epoch: {}'.format(epoch + 1))
    print('Loss: train: {:.6f}              test: {:.6f}'.format(trn_loss, tst_loss))
    print('Accuracy: train: {:.6f}          test : {:.6f}'.format(trn_acc, tst_acc))
    print("Learning time: ",  (time.time() - start_time))
    print()

    trn_loss_out.append(trn_loss)
    tst_loss_out.append(tst_loss)
    trn_acc_out.append(trn_acc)
    tst_acc_out.append(tst_acc)


optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('Learning')
#for epoch in range(15):
#   train(epoch)

#Plot_loss_accuracy(trn_loss_out, tst_loss_out, trn_acc_out, tst_acc_out)
with torch.no_grad():
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    output = model(torch.from_numpy(x_test.reshape(-1, 1, 28, 28).astype(np.float32)))
    output = output.detach().numpy()

    print(output[0])
    output = np.array([np.argmax(item) for item in output])

#out_right(output, x_test, y_test)
for i in range(10):
     print("Number {} accuracy: {}".format(i, np.sum(output[y_test == i] == i) / np.sum(y_test == i)))
print("Letter accuracy: {} ".format(np.sum(output[y_test == 10] == 10) / np.sum(y_test == 10)))
print("\n metrics.classification_report")
print(metrics.classification_report(y_test, output))


for i in range(11):
    list_ind = np.where(y_test == i)
    wrong_size.append(np.sum(y_test[list_ind] != output[list_ind]))
    for j in list_ind[0]:
        if output[j] != y_test[j]:
            wrong_index.append(j)
            break

for i in range(11):
    plt.subplot(4, 3, i + 1)
    img = x_test[wrong_index[i]]
    real = y_test[wrong_index[i]]
    if (output[wrong_index[i]] == 10):
        if (real==10):
                real = 'letter'
        ind = 'letter / {}'.format(real)
    else:
        if (real==10):
                real = 'letter'
        ind = '{} / {}'.format(output[wrong_index[i]], real)

    plt.title(ind)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.axis('off')

plt.subplots_adjust(hspace=0.5)
plt.show()
out_list = [(i, wrong_index[i], wrong_size[i]) for i in range(11)]
#print(out_list)
out_list.sort(key=lambda x: x[2])
for i in range(11):
    print('number: {}, wrong index: {}, wrong Size: {}'.format(out_list[i][0], out_list[i][1], out_list[i][2]))
np.sum(y_test == output) / y_test.shape[0]






