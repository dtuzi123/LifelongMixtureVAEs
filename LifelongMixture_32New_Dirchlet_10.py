import tensorflow as tf
import mnist_data

import tensorflow.contrib.slim as slim
import time
import seaborn as sns
from Assign_Dataset import *
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from Support import *
from Mnist_DataHandle import *
from HSICSupport import *
from scipy.misc import imsave as ims
from utils import *
from glob import glob

import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import os, gzip

#from Basic_structure import *
from data_hand import *
from CIFAR10 import *

#os.environ['CUDA_VISIBLE_DEVICES']='0'

distributions = tf.distributions
from Mixture_Models import *
import keras.datasets.cifar10 as cifar10

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True

def file_name(file_dir):
    t1 = []
    file_dir = "E:/Third_Experiment/Multiple_GAN_codes/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "E:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "E:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)

        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, task_state, Component_Count):
    lossTotal = 0
    count = 4
    Final_totalScore = 0
    # for t1 in range(count):

    isUsed = False

    # encoding
    mu1, sigma1 = Encoder_SVHN(x_hat, "encoder1", reuse=isUsed)
    mu2, sigma2 = Encoder_SVHN(x_hat, "encoder2", reuse=isUsed)
    mu3, sigma3 = Encoder_SVHN(x_hat, "encoder3", reuse=isUsed)
    mu4, sigma4 = Encoder_SVHN(x_hat, "encoder4", reuse=isUsed)
    mu5, sigma5 = Encoder_SVHN(x_hat, "encoder5", reuse=isUsed)
    mu6, sigma6 = Encoder_SVHN(x_hat, "encoder6", reuse=isUsed)
    mu7, sigma7 = Encoder_SVHN(x_hat, "encoder7", reuse=isUsed)
    mu8, sigma8 = Encoder_SVHN(x_hat, "encoder8", reuse=isUsed)
    mu9, sigma9 = Encoder_SVHN(x_hat, "encoder9", reuse=isUsed)
    mu10, sigma8 = Encoder_SVHN(x_hat, "encoder10", reuse=isUsed)


    z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1), 0, 1, dtype=tf.float32)
    z2 = mu2 + sigma2 * tf.random_normal(tf.shape(mu2), 0, 1, dtype=tf.float32)
    z3 = mu3 + sigma3 * tf.random_normal(tf.shape(mu3), 0, 1, dtype=tf.float32)
    z4 = mu4 + sigma4 * tf.random_normal(tf.shape(mu4), 0, 1, dtype=tf.float32)
    z5 = mu5 + sigma5 * tf.random_normal(tf.shape(mu5), 0, 1, dtype=tf.float32)
    z6 = mu6 + sigma6 * tf.random_normal(tf.shape(mu6), 0, 1, dtype=tf.float32)
    z7 = mu7 + sigma7 * tf.random_normal(tf.shape(mu7), 0, 1, dtype=tf.float32)
    z8 = mu8 + sigma8 * tf.random_normal(tf.shape(mu8), 0, 1, dtype=tf.float32)
    z9 = mu9 + sigma8 * tf.random_normal(tf.shape(mu9), 0, 1, dtype=tf.float32)
    z10 = mu10 + sigma8 * tf.random_normal(tf.shape(mu10), 0, 1, dtype=tf.float32)

    x1 = Generator_SVHN(z1, "decoder1", reuse=isUsed)
    x2 = Generator_SVHN(z2, "decoder2", reuse=isUsed)
    x3 = Generator_SVHN(z3, "decoder3", reuse=isUsed)
    x4 = Generator_SVHN(z4, "decoder4", reuse=isUsed)
    x5 = Generator_SVHN(z5, "decoder5", reuse=isUsed)
    x6 = Generator_SVHN(z6, "decoder6", reuse=isUsed)
    x7 = Generator_SVHN(z7, "decoder7", reuse=isUsed)
    x8 = Generator_SVHN(z8, "decoder8", reuse=isUsed)
    x9 = Generator_SVHN(z9, "decoder9", reuse=isUsed)
    x10 = Generator_SVHN(z10, "decoder10", reuse=isUsed)

    s1_1 = tf.reshape(x1, (-1, 32 * 32 * 3)) * task_state[:, 0:1]
    s2_1 = tf.reshape(x2, (-1, 32 * 32 * 3)) * task_state[:, 1:2]
    s3_1 = tf.reshape(x3, (-1, 32 * 32 * 3)) * task_state[:, 2:3]
    s4_1 = tf.reshape(x4, (-1, 32 * 32 * 3)) * task_state[:, 3:4]
    s5_1 = tf.reshape(x5, (-1, 32 * 32 * 3)) * task_state[:, 4:5]
    s6_1 = tf.reshape(x6, (-1, 32 * 32 * 3)) * task_state[:, 5:6]
    s7_1 = tf.reshape(x7, (-1, 32 * 32 * 3)) * task_state[:, 6:7]
    s8_1 = tf.reshape(x8, (-1, 32 * 32 * 3)) * task_state[:, 7:8]
    s9_1 = tf.reshape(x9, (-1, 32 * 32 * 3)) * task_state[:, 8:9]
    s10_1 = tf.reshape(x10, (-1, 32 * 32 * 3)) * task_state[:, 9:10]

    reco = s1_1 + s2_1 + s3_1 + s4_1 + s5_1+s6_1+s7_1+s8_1+s9_1+s10_1
    reco = reco / (task_state[0, 0] + task_state[0, 1] + task_state[0, 2] + task_state[0, 3] + task_state[0, 4] + task_state[0, 5]+task_state[0, 6]+task_state[0, 7]+task_state[0, 8]+task_state[0, 9])
    reco = tf.reshape(reco, (-1, 32, 32, 3))

    # Calculate task relationship

    reco1 = tf.reduce_mean(tf.reduce_sum(tf.square(x1 - x_hat), [1, 2, 3]))
    reco2 = tf.reduce_mean(tf.reduce_sum(tf.square(x2 - x_hat), [1, 2, 3]))
    reco3 = tf.reduce_mean(tf.reduce_sum(tf.square(x3 - x_hat), [1, 2, 3]))
    reco4 = tf.reduce_mean(tf.reduce_sum(tf.square(x4 - x_hat), [1, 2, 3]))
    reco5 = tf.reduce_mean(tf.reduce_sum(tf.square(x5 - x_hat), [1, 2, 3]))
    reco6 = tf.reduce_mean(tf.reduce_sum(tf.square(x6 - x_hat), [1, 2, 3]))
    reco7 = tf.reduce_mean(tf.reduce_sum(tf.square(x7 - x_hat), [1, 2, 3]))
    reco8 = tf.reduce_mean(tf.reduce_sum(tf.square(x8 - x_hat), [1, 2, 3]))
    reco9 = tf.reduce_mean(tf.reduce_sum(tf.square(x9 - x_hat), [1, 2, 3]))
    reco10 = tf.reduce_mean(tf.reduce_sum(tf.square(x10 - x_hat), [1, 2, 3]))

    reco1_ = reco1 + (1 - task_state[0, 0]) * 1000000
    reco2_ = reco2 + (1 - task_state[0, 1]) * 1000000
    reco3_ = reco3 + (1 - task_state[0, 2]) * 1000000
    reco4_ = reco4 + (1 - task_state[0, 3]) * 1000000
    reco5_ = reco5 + (1 - task_state[0, 4]) * 1000000
    reco6_ = reco6 + (1 - task_state[0, 5]) * 1000000
    reco7_ = reco7 + (1 - task_state[0, 6]) * 1000000
    reco8_ = reco8 + (1 - task_state[0, 7]) * 1000000
    reco9_ = reco9 + (1 - task_state[0, 8]) * 1000000
    reco10_ = reco10 + (1 - task_state[0, 9]) * 1000000

    totalScore = tf.stack((reco1_, reco2_, reco3_, reco4_,reco5_,reco6_,reco7_,reco8_,reco9_,reco10_), axis=0)
    Final_totalScore = totalScore

    mixParameter = task_state[0]
    sum = mixParameter[0] +mixParameter[1]+mixParameter[2]+mixParameter[3]+mixParameter[4]+mixParameter[5]+mixParameter[6]+mixParameter[7]+mixParameter[8]+mixParameter[9]
    mixParameter = mixParameter / sum

    dist = tf.distributions.Dirichlet(mixParameter)
    mix_samples = dist.sample()

    reco1_loss = reco1 * mix_samples[0]
    reco2_loss = reco2 * mix_samples[1]
    reco3_loss = reco3 * mix_samples[2]
    reco4_loss = reco4 * mix_samples[3]
    reco5_loss = reco5 * mix_samples[4]
    reco6_loss = reco6 * mix_samples[5]
    reco7_loss = reco7 * mix_samples[6]
    reco8_loss = reco8 * mix_samples[7]
    reco9_loss = reco9 * mix_samples[8]
    reco10_loss = reco10 * mix_samples[9]

    # loss
    marginal_likelihood = (reco1_loss + reco2_loss + reco3_loss + reco4_loss+reco5_loss+reco6_loss+reco7_loss+reco8_loss+reco9_loss+reco10_loss)

    k1 = 0.5 * tf.reduce_sum(
        tf.square(mu1) + tf.square(sigma1) - tf.log(1e-8 + tf.square(sigma1)) - 1,
        1)
    k2 = 0.5 * tf.reduce_sum(
        tf.square(mu2) + tf.square(sigma2) - tf.log(1e-8 + tf.square(sigma2)) - 1,
        1)
    k3 = 0.5 * tf.reduce_sum(
        tf.square(mu3) + tf.square(sigma3) - tf.log(1e-8 + tf.square(sigma3)) - 1,
        1)
    k4 = 0.5 * tf.reduce_sum(
        tf.square(mu4) + tf.square(sigma4) - tf.log(1e-8 + tf.square(sigma4)) - 1,
        1)
    k5 = 0.5 * tf.reduce_sum(
        tf.square(mu5) + tf.square(sigma5) - tf.log(1e-8 + tf.square(sigma5)) - 1,
        1)
    k6 = 0.5 * tf.reduce_sum(
        tf.square(mu6) + tf.square(sigma6) - tf.log(1e-8 + tf.square(sigma6)) - 1,
        1)
    k7 = 0.5 * tf.reduce_sum(
        tf.square(mu7) + tf.square(sigma7) - tf.log(1e-8 + tf.square(sigma7)) - 1,
        1)
    k8 = 0.5 * tf.reduce_sum(
        tf.square(mu8) + tf.square(sigma8) - tf.log(1e-8 + tf.square(sigma8)) - 1,
        1)
    k9 = 0.5 * tf.reduce_sum(
        tf.square(mu9) + tf.square(sigma9) - tf.log(1e-8 + tf.square(sigma9)) - 1,
        1)
    k10 = 0.5 * tf.reduce_sum(
        tf.square(mu10) + tf.square(sigma10) - tf.log(1e-8 + tf.square(sigma10)) - 1,
        1)

    k1 = tf.reduce_mean(k1)
    k2 = tf.reduce_mean(k2)
    k3 = tf.reduce_mean(k3)
    k4 = tf.reduce_mean(k4)
    k5 = tf.reduce_mean(k5)
    k6 = tf.reduce_mean(k6)
    k7 = tf.reduce_mean(k7)
    k8 = tf.reduce_mean(k8)
    k9 = tf.reduce_mean(k9)
    k10 = tf.reduce_mean(k10)

    KL_divergence = k1 * mix_samples[0] + k2 * mix_samples[1] + k3 * mix_samples[2] + k4 * mix_samples[3]+ k5 * mix_samples[4]+ k6 * mix_samples[5]+k7 * mix_samples[6]+k8 * mix_samples[7]+k9 * mix_samples[8]+k10 * mix_samples[9]
    KL_divergence = KL_divergence

    p2 = 1

    loss = marginal_likelihood + KL_divergence * p2
    # lossTotal = loss + lossTotal

    # loss = lossTotal / count

    z = z1
    y = reco

    return y, z, loss, -marginal_likelihood, KL_divergence, Final_totalScore


def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y


n_hidden = 500
IMAGE_SIZE_MNIST = 28
dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image

dim_z = 50

# train
n_epochs = 100
batch_size = 64
learn_rate = 0.001

train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
n_samples = train_size
# input placeholders

# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[64, 32, 32, 3], name='input_img')
x = tf.placeholder(tf.float32, shape=[64, 32, 32, 3], name='target_img')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
task_state = tf.placeholder(tf.float32, shape=[64, 10])
Component_Count = tf.placeholder(tf.float32)

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence, totalScore = autoencoder(x_hat, x, dim_img, dim_z,
                                                                             n_hidden, keep_prob,
                                                                             task_state, Component_Count)

# optimization
t_vars = tf.trainable_variables()
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, var_list=t_vars)

# train

total_batch = int(n_samples / batch_size)

min_tot_loss = 1e99
ADD_NOISE = False

train_data2_ = train_total_data[:, :-mnist_data.NUM_LABELS]
train_y = train_total_data[:, 784:784 + mnist_data.NUM_LABELS]

# MNIST dataset   load datasets
mnistName = "mnist"
fashionMnistName = "Fashion"

mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

mnist_train_x = mnist_train_x
mnist_train_y = np.zeros((np.shape(x_train)[0], 4))
mnist_train_y[:, 0] = 1
mnist_label = mnist_train_label
mnist_label_test = mnist_label_test
mnist_test_x = mnist_test
mnist_test_y = mnist_label_test

svhn_train_x = x_train
svhn_train_y = np.zeros((np.shape(x_train)[0], 4))
svhn_train_y[:, 0] = 1
svhn_label = y_train
svhn_label_test = y_test
svhn_test_x = x_test
svhn_test_y = y_test

fashion_train_x, fashion_train_label, fashion_test_x, fashion_test_label = GiveFashion_32()

cifar_train_x, cifar_train_label, cifar_test_x, cifar_test_label = prepare_data()
cifar_train_x, cifar_test_x = color_preprocessing(cifar_train_x, cifar_test_x)
bc = 0
'''
files = file_name(1)
data_files = files
data_files = sorted(data_files)
data_files = np.array(data_files)  # for tl.iterate.minibatches
n_examples = np.shape(data_files)[0]
batch = [get_image(batch_file, 105, 105,
                   resize_height=28, resize_width=28,
                   crop=False, grayscale=True) \
         for batch_file in data_files]
thirdX = np.array(batch)

for t1 in range(n_examples):
    a1 = thirdX[t1]
    for p1 in range(28):
        for p2 in range(28):
            if thirdX[t1, p1, p2] == 1.0:
                thirdX[t1, p1, p2] = 0
            else:
                thirdX[t1, p1, p2] = 1

thirdX = np.reshape(thirdX, (-1, 28, 28, 1))
t1 = 15000
third_x_train = thirdX[0:t1]
third_x_test = thirdX[t1:np.shape(thirdX)[0]]

bc = thirdX
'''


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str


def softmax(x):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s


def CalculateMSE(x1, x2):
    c = tf.square(x1 - x2)
    return tf.reduce_mean(c)

def MyTest():
    saver = tf.train.Saver()

    stateState = np.zeros((batch_size,10))
    stateState[:, 0] = 1
    stateState[:, 1] = 1
    stateState[:, 2] = 1
    stateState[:, 3] = 1
    stateState[:, 4] = 1
    stateState[:, 5] = 1

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})
        saver.restore(sess, 'models/LifelongMixture_32New_Dirichlet_10')
        currentTask = 4

        for taskIndex in range(currentTask):
            x_fixed1 = mnist_test_x[0:batch_size]
            x_fixed2 = fashion_test_x[0:batch_size]
            x_fixed3 = svhn_test_x[0:batch_size]
            x_fixed4 = cifar_test_x[0:batch_size]

            if taskIndex == 0:
                x_fixed = x_fixed1
                x_test = mnist_test_x
            elif taskIndex == 1:
                x_fixed = x_fixed2
                x_test = fashion_test_x
            elif taskIndex == 2:
                x_fixed = x_fixed3
                x_test = svhn_test_x
            elif taskIndex == 3:
                x_fixed = x_fixed4
                x_test = cifar_test_x

            score = sess.run(totalScore, feed_dict={x_hat: x_fixed, keep_prob: 1,task_state:stateState})
            a = np.argmin(score, axis=0)
            index = a

            if index == 0:
                mu1, sigma1 = Encoder_SVHN(x_hat, "encoder1",reuse=True)
                z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1), 0, 1, dtype=tf.float32)
                reco = Generator_SVHN(z1,"decoder1",reuse=True)
            elif index == 1:
                mu2, sigma2 = Encoder_SVHN(x_hat, "encoder2", reuse=True)
                z2 = mu2 + sigma2 * tf.random_normal(tf.shape(mu2), 0, 1, dtype=tf.float32)
                reco = Generator_SVHN(z2, "decoder2", reuse=True)
            elif index == 2:
                mu3, sigma3 = Encoder_SVHN(x_hat, "encoder3", reuse=True)
                z3 = mu3 + sigma3 * tf.random_normal(tf.shape(mu3), 0, 1, dtype=tf.float32)
                reco = Generator_SVHN(z3, "decoder3", reuse=True)
            elif index == 3:
                mu4, sigma4 = Encoder_SVHN(x_hat, "encoder4", reuse=True)
                z4 = mu4 + sigma4 * tf.random_normal(tf.shape(mu4), 0, 1, dtype=tf.float32)
                reco = Generator_SVHN(z4, "decoder4", reuse=True)

            reco_error = tf.reduce_mean(tf.reduce_sum(tf.square(reco - x_hat), [1, 2, 3]))

            myIndex = int(np.shape(x_test)[0] / batch_size)
            myArr = []
            sumError = 0

            x_in1 = tf.placeholder(tf.float32, shape=[None, 28 * 28])
            x_in2 = tf.placeholder(tf.float32, shape=[None, 28 * 28])

            x_in1 = tf.placeholder(tf.float32, shape=[None, 32 * 32*3])
            x_in2 = tf.placeholder(tf.float32, shape=[None, 32 * 32*3])

            '''
            myReal = []
            myArr2 = []
            totalError = 0
            for k in range(myIndex):
                xx = x_test[k * batch_size:(k + 1) * batch_size]
                outputs = sess.run(reco, feed_dict={x_hat: xx, keep_prob: 1})
                errors = sess.run(reco_error, feed_dict={x_hat: xx, keep_prob: 1})
                totalError = totalError + errors
                for j in range(64):
                    a1 = outputs[j]
                    image = cv2.normalize(a1, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                    image2 = np.transpose(image, (2, 0, 1))
                    myArr.append(image2)
                    myArr2.append(image)
                    myReal.append(xx[j])

            totalError = totalError / myIndex

            myArr = myArr[0:5000]

            score = get_inception_score(myArr)

            avgError = sumError / myIndex
            '''

            myReco = sess.run(reco, feed_dict={x_hat: x_fixed, keep_prob: 1})
            y_RPR = np.reshape(myReco, (-1, 32, 32,3))

            real = np.reshape(x_fixed, (-1, 32, 32,3))
            ims("results/" + "real" + str(taskIndex) + "_mini.png", merge2(real[:16], [2, 8]))
            ims("results/" + "reco" + str(taskIndex) + "_mini.png", merge2(y_RPR[:16], [2, 8]))

            bc = 0

    bc = 0

#MyTest()


x_train = mnist_train_x
x_test = mnist_test_x

x_fixed = train_data2_[0:batch_size]
saver = tf.train.Saver()

isWeight = False
currentTask = 4
isWeight = False

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

    if isWeight:
        saver.restore(sess, 'models/LifelongMixture_32New_Dirichlet_10')

    else:
        # training

        n_epochs = 100

        stateState = np.zeros((batch_size, 10))
        stateState[:, 0] = 1
        stateState[:, 1] = 1
        stateState[:, 2] = 1
        stateState[:, 3] = 1
        stateState[:, 4] = 1
        stateState[:, 5] = 1
        stateState[:, 6] = 1
        stateState[:, 7] = 1
        stateState[:, 8] = 1
        stateState[:, 9] = 1

        for taskIndex in range(currentTask):

            if taskIndex == 0:
                x_train = mnist_train_x
                x_fixed = x_train[0:batch_size]
            elif taskIndex == 1:
                x_train = fashion_train_x
                x_fixed = x_train[0:batch_size]
            elif taskIndex == 2:
                x_train = svhn_train_x
                x_fixed = x_train[0:batch_size]
            elif taskIndex == 3:
                x_train = cifar_train_x
                x_fixed = x_train[0:batch_size]

            n_samples = np.shape(x_train)[0]
            total_batch = int(n_samples / batch_size)

            for epoch in range(n_epochs):
                # Random shuffling
                index = [i for i in range(np.shape(x_train)[0])]
                random.shuffle(index)
                x_train = x_train[index]

                # Loop over all batches
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    batch_xs_input = x_train[i * batch_size:i * batch_size + batch_size]
                    batch_xs_target = batch_xs_input

                    if ADD_NOISE:
                        batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                        batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                    _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                        (train_op, loss, neg_marginal_likelihood, KL_divergence),
                        feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 1.0, task_state: stateState})

                    print("epoch %f: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                        epoch, tot_loss, loss_likelihood, loss_divergence))

                y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState})
                y_RPR = np.reshape(y_PRR, (-1, 32, 32, 3))
                ims("results/" + "VAE" + str(epoch) + ".jpg", merge2(y_RPR[:64], [8, 8]))

                if epoch > 0:
                    x_fixed_image = np.reshape(x_fixed, (-1, 32, 32, 3))
                    ims("results/" + "Real" + str(epoch) + ".jpg", merge2(x_fixed_image[:64], [8, 8]))

            # select the most relevant component
            score = sess.run(totalScore, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState})
            a = np.argmin(score, axis=0)
            index = a
            if index == 0:
                stateState[:, 0:1] = 0
            elif index == 1:
                stateState[:, 1:2] = 0
            elif index == 2:
                stateState[:, 2:3] = 0
            elif index == 3:
                stateState[:, 3:4] = 0
            elif index == 4:
                stateState[:, 4:5] = 0
            elif index == 5:
                stateState[:, 5:6] = 0
            elif index == 6:
                stateState[:, 6:7] = 0
            elif index == 7:
                stateState[:, 7:8] = 0
            elif index == 8:
                stateState[:, 8:9] = 0
            elif index == 9:
                stateState[:, 9:10] = 0

    saver.save(sess, 'models/LifelongMixture_32New_Dirichlet_10')
