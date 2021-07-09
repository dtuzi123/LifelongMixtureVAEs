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

from data_hand import *
from CIFAR10 import *

distributions = tf.distributions
from Mixture_Models import *

os.environ['CUDA_VISIBLE_DEVICES']='0'

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
def autoencoder(x_hat, x,y_in, dim_img, dim_z, n_hidden, keep_prob, task_state, Component_Count):
    # encoding
    mu1, sigma1 = Encoder_SVHN2(x_hat, "encoder1")
    logits1, softmaxValue1 = Encoder_SVHN_Classifier(x_hat, "classifier1")
    mu2, sigma2 = Encoder_SVHN2(x_hat, "encoder2")
    logits2, softmaxValue2 = Encoder_SVHN_Classifier(x_hat, "classifier2")
    mu3, sigma3 = Encoder_SVHN2(x_hat, "encoder3")
    logits3, softmaxValue3 = Encoder_SVHN_Classifier(x_hat, "classifier3")
    mu4, sigma4 = Encoder_SVHN2(x_hat, "encoder4")
    logits4, softmaxValue4 = Encoder_SVHN_Classifier(x_hat, "classifier4")

    log_y1 = tf.log(softmaxValue1 + 1e-10)
    discrete1 = my_gumbel_softmax_sample(log_y1, np.arange(10))

    log_y2 = tf.log(softmaxValue2 + 1e-10)
    discrete2 = my_gumbel_softmax_sample(log_y2, np.arange(10))

    log_y3 = tf.log(softmaxValue3 + 1e-10)
    discrete3 = my_gumbel_softmax_sample(log_y3, np.arange(10))

    log_y4 = tf.log(softmaxValue4 + 1e-10)
    discrete4 = my_gumbel_softmax_sample(log_y4, np.arange(10))

    z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1), 0, 1, dtype=tf.float32)
    z2 = mu2 + sigma2 * tf.random_normal(tf.shape(mu2), 0, 1, dtype=tf.float32)
    z3 = mu3 + sigma3 * tf.random_normal(tf.shape(mu3), 0, 1, dtype=tf.float32)
    z4 = mu4 + sigma4 * tf.random_normal(tf.shape(mu4), 0, 1, dtype=tf.float32)

    code1 = tf.concat((z1,discrete1),axis=1)
    code2 = tf.concat((z2,discrete2),axis=1)
    code3 = tf.concat((z3,discrete3),axis=1)
    code4 = tf.concat((z4,discrete4),axis=1)

    s1 = Generator_SharedSVHN(code1, "sharedDecoder", reuse=False)
    s2 = Generator_SharedSVHN(code2, "sharedDecoder", reuse=True)
    s3 = Generator_SharedSVHN(code3, "sharedDecoder", reuse=True)
    s4 = Generator_SharedSVHN(code4, "sharedDecoder", reuse=True)

    x1 = Generator_SubSVHN(s1, "decoder1", reuse=False)
    x2 = Generator_SubSVHN(s2, "decoder2", reuse=False)
    x3 = Generator_SubSVHN(s3, "decoder3", reuse=False)
    x4 = Generator_SubSVHN(s4, "decoder4", reuse=False)

    s1_1 = tf.reshape(x1, (-1, 32 * 32 * 3)) * task_state[:, 0:1]
    s2_1 = tf.reshape(x2, (-1, 32 * 32 * 3)) * task_state[:, 1:2]
    s3_1 = tf.reshape(x3, (-1, 32 * 32 * 3)) * task_state[:, 2:3]
    s4_1 = tf.reshape(x4, (-1, 32 * 32 * 3)) * task_state[:, 3:4]

    # Calculate task relationship

    reco1 = tf.reduce_mean(tf.reduce_sum(tf.square(x1 - x_hat), [1, 2, 3]))
    reco2 = tf.reduce_mean(tf.reduce_sum(tf.square(x2 - x_hat), [1, 2, 3]))
    reco3 = tf.reduce_mean(tf.reduce_sum(tf.square(x3 - x_hat), [1, 2, 3]))
    reco4 = tf.reduce_mean(tf.reduce_sum(tf.square(x4 - x_hat), [1, 2, 3]))

    reco1_ = reco1 + (1-task_state[0,0]) * 1000000
    reco2_ = reco2 + (1-task_state[0,1]) * 1000000
    reco3_ = reco3 + (1-task_state[0,2]) * 1000000
    reco4_ = reco4 + (1-task_state[0,3]) * 1000000

    totalScore = tf.stack((reco1_,reco2_,reco3_,reco4_),axis=0)

    mixParameter = task_state[0]
    sum = mixParameter[0] + mixParameter[1] + mixParameter[2] + mixParameter[3]
    mixParameter = mixParameter / sum

    dist = tf.distributions.Dirichlet(mixParameter)
    mix_samples = dist.sample()

    # loss
    reco1_loss = reco1 * mix_samples[0]
    reco2_loss = reco2 * mix_samples[1]
    reco3_loss = reco3 * mix_samples[2]
    reco4_loss = reco4 * mix_samples[3]

    # loss
    marginal_likelihood = (reco1_loss + reco2_loss + reco3_loss + reco4_loss)

    # clasification loss
    classLoss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=y_in))
    classLoss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=y_in))
    classLoss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=y_in))
    classLoss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits4, labels=y_in))
    totalLogits = (classLoss1*task_state[0,0] + classLoss2*task_state[0,1] +classLoss3*task_state[0,2] + classLoss4*task_state[0,3])/(task_state[0,0] +task_state[0,1] + task_state[0,2] +task_state[0,3])
    classLoss = totalLogits

    classloss1_ = classLoss1 + (1 - task_state[0, 0]) * 1000000
    classloss2_ = classLoss2 + (1 - task_state[0, 1]) * 1000000
    classloss3_ = classLoss3 + (1 - task_state[0, 2]) * 1000000
    classloss4_ = classLoss4 + (1 - task_state[0, 3]) * 1000000

    totalScore2 = tf.stack((classloss1_, classloss2_, classloss3_, classloss4_), axis=0)

    #KL divergence between Gaussian distributions
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

    k1 = tf.reduce_mean(k1)
    k2 = tf.reduce_mean(k2)
    k3 = tf.reduce_mean(k3)
    k4 = tf.reduce_mean(k4)

    KL_divergence = k1 * mix_samples[0] + k2 * mix_samples[1] + k3 * mix_samples[2] + k4 * mix_samples[3]
    KL_divergence = KL_divergence

    #KL divergence between Gumble softmax distributions
    # KL divergence on gumble softmax
    KL_y1 = tf.reduce_mean(tf.reduce_sum(softmaxValue1 * (tf.log(softmaxValue1 + 1e-10) - tf.log(1.0 / 10.0)), 1))
    KL_y2 = tf.reduce_mean(tf.reduce_sum(softmaxValue2 * (tf.log(softmaxValue2 + 1e-10) - tf.log(1.0 / 10.0)), 1))
    KL_y3 = tf.reduce_mean(tf.reduce_sum(softmaxValue3 * (tf.log(softmaxValue3 + 1e-10) - tf.log(1.0 / 10.0)), 1))
    KL_y4 = tf.reduce_mean(tf.reduce_sum(softmaxValue4 * (tf.log(softmaxValue4 + 1e-10) - tf.log(1.0 / 10.0)), 1))
    KL_Y = KL_y1 * task_state[0, 0] + KL_y2 * task_state[0,1] + KL_y3 * task_state[0,2] + KL_y4 * task_state[0,3]
    KL_Y = KL_Y / (task_state[0, 0] + task_state[0, 1] + task_state[0, 2] + task_state[0, 3])

    p1 = 1
    p2 = 0.0001
    loss = marginal_likelihood + KL_divergence * p1 #+ KL_Y*p2

    z = z1
    y = s1_1

    return y, z, loss, -marginal_likelihood, KL_divergence,totalScore,classLoss,totalScore2


def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y

def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

#开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

def MyTest():
    saver = tf.train.Saver()

    stateState = np.zeros((batch_size, 4))
    stateState[:, 0] = 1
    stateState[:, 1] = 1
    stateState[:, 2] = 1
    stateState[:, 3] = 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})
        saver.restore(sess, 'models/Mixture4_Lifelong32_conditional')

        x_fixed1 = mnist_train_x[0:batch_size]
        x_fixed2 = fashion_train_x[0:batch_size]
        x_fixed3 = svhn_train_x[0:batch_size]
        x_fixed4 = cifar_train_x[0:batch_size]

        x_fixed = x_fixed4
        score = sess.run(totalScore, feed_dict={x_hat: x_fixed, keep_prob: 1,task_state:stateState})
        a = np.argmin(score, axis=0)
        index = a
        predictLabel = 0

        test_x = cifar_test_x
        test_labels = cifar_test_label

        #index = 2
        if index == 0:
            mu1, sigma1 = Encoder_SVHN2(x_hat, "encoder1",reuse=True)
            logits1, softmaxValue1 = Encoder_SVHN_Classifier(x_hat, "classifier1",reuse=True)

            log_y1 = tf.log(softmaxValue1 + 1e-10)
            discrete1 = my_gumbel_softmax_sample(log_y1, np.arange(10))
            z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1), 0, 1, dtype=tf.float32)
            code1 = tf.concat((z1, discrete1), axis=1)
            s1 = Generator_SVHN(code1, "decoder1",reuse=True)
            reco = s1
            predictLabel = softmaxValue1

        elif index == 1:
            mu2, sigma2 = Encoder_SVHN2(x_hat, "encoder2", reuse=True)
            logits2, softmaxValue2 = Encoder_SVHN_Classifier(x_hat, "classifier2", reuse=True)

            log_y2 = tf.log(softmaxValue2 + 1e-10)
            discrete2 = my_gumbel_softmax_sample(log_y2, np.arange(10))
            z2 = mu2 + sigma2 * tf.random_normal(tf.shape(mu2), 0, 1, dtype=tf.float32)
            code2 = tf.concat((z2, discrete2), axis=1)
            s2 = Generator_SVHN(code2, "decoder2",reuse=True)
            reco = s2
            predictLabel = softmaxValue2

        elif index == 2:
            mu3, sigma3 = Encoder_SVHN2(x_hat, "encoder3", reuse=True)
            logits3, softmaxValue3 = Encoder_SVHN_Classifier(x_hat, "classifier3", reuse=True)

            log_y3 = tf.log(softmaxValue3 + 1e-10)
            discrete3 = my_gumbel_softmax_sample(log_y3, np.arange(10))
            z3 = mu3 + sigma3 * tf.random_normal(tf.shape(mu3), 0, 1, dtype=tf.float32)
            code3 = tf.concat((z3, discrete3), axis=1)
            s3 = Generator_SVHN(code3, "decoder3",reuse=True)
            reco = s3
            predictLabel = softmaxValue3

        elif index == 3:
            mu4, sigma4 = Encoder_SVHN2(x_hat, "encoder4", reuse=True)
            logits4, softmaxValue4 = Encoder_SVHN_Classifier(x_hat, "classifier4", reuse=True)

            log_y4 = tf.log(softmaxValue4 + 1e-10)
            discrete4 = my_gumbel_softmax_sample(log_y4, np.arange(10))
            z4 = mu4 + sigma4 * tf.random_normal(tf.shape(mu4), 0, 1, dtype=tf.float32)
            code4 = tf.concat((z4, discrete4), axis=1)
            s4 = Generator_SVHN(code4, "decoder4",reuse=True)
            reco = s4
            predictLabel = softmaxValue4

        myArr = []
        for t in range(int(np.shape(test_x)[0]/batch_size)):
            batchX = test_x[t*batch_size:(t+1)*batch_size]
            batchY = test_labels[t*batch_size:(t+1)*batch_size]
            pre = sess.run(predictLabel, feed_dict={x_hat: batchX, keep_prob: 1,task_state:stateState})
            for t1 in range(batch_size):
                myArr.append(pre[t1])

        myArr = np.array(myArr)
        myArr = props_to_onehot(myArr)

        accurateCount = 0
        for tindex in range(np.shape(myArr)[0]):
            isCorrect = False
            for t2Index in range(10):
                if myArr[tindex,t2Index] == 1 and test_labels[tindex,t2Index] == 1:
                    isCorrect=True
                    break

            if isCorrect == True:
                accurateCount = accurateCount+1

        accuracy = accurateCount / np.shape(myArr)[0]

        myReco = sess.run(reco, feed_dict={x_hat: x_fixed, keep_prob: 1,task_state:stateState})
        y_RPR = np.reshape(myReco, (-1, 32, 32,3))

        real = np.reshape(x_fixed, (-1, 32, 32,3))
        ims("results/" + "rea" + str(3) + ".jpg", merge2(real[:64], [8, 8]))
        ims("results/" + "H" + str(3) + ".jpg", merge2(y_RPR[:64], [8, 8]))

        bc = 0

def CalculateAccuracy(sess,x_test,test_labels):
    saver = tf.train.Saver()

    stateState2 = np.zeros((batch_size, 4))
    stateState2[:, 0] = 1
    stateState2[:, 1] = 1
    stateState2[:, 2] = 1
    stateState2[:, 3] = 1

    x_fixed1 = mnist_train_x[0:batch_size]
    x_fixed2 = fashion_train_x[0:batch_size]
    x_fixed3 = svhn_train_x[0:batch_size]
    x_fixed4 = cifar_train_x[0:batch_size]

    x_fixed = x_test[0:batch_size]
    score = sess.run(totalScore, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState2})
    a = np.argmin(score, axis=0)
    index = a
    predictLabel = 0

    if index == 0:
        mu1, sigma1 = Encoder_SVHN2(x_hat, "encoder1", reuse=True)
        logits1, softmaxValue1 = Encoder_SVHN_Classifier(x_hat, "classifier1", reuse=True)

        log_y1 = tf.log(softmaxValue1 + 1e-10)
        discrete1 = my_gumbel_softmax_sample(log_y1, np.arange(10))
        z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1), 0, 1, dtype=tf.float32)
        code1 = tf.concat((z1, discrete1), axis=1)

        s1 = Generator_SharedSVHN(code1, "sharedDecoder", reuse=True)
        x1 = Generator_SubSVHN(s1, "decoder1", reuse=True)
        reco = x1
        predictLabel = softmaxValue1

    elif index == 1:
        mu2, sigma2 = Encoder_SVHN2(x_hat, "encoder2", reuse=True)
        logits2, softmaxValue2 = Encoder_SVHN_Classifier(x_hat, "classifier2", reuse=True)

        log_y2 = tf.log(softmaxValue2 + 1e-10)
        discrete2 = my_gumbel_softmax_sample(log_y2, np.arange(10))
        z2 = mu2 + sigma2 * tf.random_normal(tf.shape(mu2), 0, 1, dtype=tf.float32)
        code2 = tf.concat((z2, discrete2), axis=1)
        s2 = Generator_SharedSVHN(code2, "sharedDecoder", reuse=True)
        x2 = Generator_SubSVHN(s2, "decoder2", reuse=True)
        reco = x2
        predictLabel = softmaxValue2

    elif index == 2:
        mu3, sigma3 = Encoder_SVHN2(x_hat, "encoder3", reuse=True)
        logits3, softmaxValue3 = Encoder_SVHN_Classifier(x_hat, "classifier3", reuse=True)

        log_y3 = tf.log(softmaxValue3 + 1e-10)
        discrete3 = my_gumbel_softmax_sample(log_y3, np.arange(10))
        z3 = mu3 + sigma3 * tf.random_normal(tf.shape(mu3), 0, 1, dtype=tf.float32)
        code3 = tf.concat((z3, discrete3), axis=1)
        s3 = Generator_SharedSVHN(code3, "sharedDecoder", reuse=True)
        x3 = Generator_SubSVHN(s3, "decoder3", reuse=True)
        predictLabel = softmaxValue3

    elif index == 3:
        mu4, sigma4 = Encoder_SVHN2(x_hat, "encoder4", reuse=True)
        logits4, softmaxValue4 = Encoder_SVHN_Classifier(x_hat, "classifier4", reuse=True)

        log_y4 = tf.log(softmaxValue4 + 1e-10)
        discrete4 = my_gumbel_softmax_sample(log_y4, np.arange(10))
        z4 = mu4 + sigma4 * tf.random_normal(tf.shape(mu4), 0, 1, dtype=tf.float32)
        code4 = tf.concat((z4, discrete4), axis=1)
        s4 = Generator_SharedSVHN(code4, "sharedDecoder", reuse=True)
        x4 = Generator_SubSVHN(s4, "decoder4", reuse=True)
        reco = s4
        predictLabel = softmaxValue4

    myArr = []
    for t in range(int(np.shape(x_test)[0] / batch_size)):
        batchX = x_test[t * batch_size:(t + 1) * batch_size]
        batchY = test_labels[t * batch_size:(t + 1) * batch_size]
        pre = sess.run(predictLabel, feed_dict={x_hat: batchX, keep_prob: 1, task_state: stateState2})
        for t1 in range(batch_size):
            myArr.append(pre[t1])

    myArr = np.array(myArr)
    myArr = props_to_onehot(myArr)

    accurateCount = 0
    for tindex in range(np.shape(myArr)[0]):
        isCorrect = False
        for t2Index in range(10):
            if myArr[tindex, t2Index] == 1 and test_labels[tindex, t2Index] == 1:
                isCorrect = True
                break

        if isCorrect == True:
            accurateCount = accurateCount + 1

    accuracy = accurateCount / np.shape(myArr)[0]

    return accuracy

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
y_in = tf.placeholder(tf.float32, shape=[64, 10])

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
task_state = tf.placeholder(tf.float32, shape=[64, 4])
Component_Count = tf.placeholder(tf.float32)

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence,totalScore,classLoss,totalScore2 = autoencoder(x_hat, x,y_in, dim_img, dim_z,
                                                                 n_hidden, keep_prob,
                                                                 task_state, Component_Count)

# optimization
T_vars = tf.trainable_variables()
encoder_vars1 = [var for var in T_vars if var.name.startswith('encoder1')]
encoder_vars2 = [var for var in T_vars if var.name.startswith('encoder2')]
encoder_vars3 = [var for var in T_vars if var.name.startswith('encoder3')]
encoder_vars4 = [var for var in T_vars if var.name.startswith('encoder4')]
decoder_vars1 = [var for var in T_vars if var.name.startswith('decoder1')]
decoder_vars2 = [var for var in T_vars if var.name.startswith('decoder2')]
decoder_vars3 = [var for var in T_vars if var.name.startswith('decoder3')]
decoder_vars4 = [var for var in T_vars if var.name.startswith('decoder4')]

classifier_vars1 = [var for var in T_vars if var.name.startswith('classifier1')]
classifier_vars2 = [var for var in T_vars if var.name.startswith('classifier2')]
classifier_vars3 = [var for var in T_vars if var.name.startswith('classifier3')]
classifier_vars4 = [var for var in T_vars if var.name.startswith('classifier4')]

classifierVars = classifier_vars1 + classifier_vars2 + classifier_vars3 + classifier_vars4
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, var_list=T_vars)
train_classifier = tf.train.AdamOptimizer(learn_rate).minimize(classLoss, var_list=classifierVars)

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

fashion_train_x,fashion_train_label,fashion_test_x,fashion_test_label = GiveFashion_32()

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

x_train = mnist_train_x
x_test = mnist_test_x

x_fixed = train_data2_[0:batch_size]
saver = tf.train.Saver()

isWeight = True
currentTask = 4

#MyTest()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

    if isWeight:
        saver.restore(sess, 'models/LifelongMixture_32New_Dirchlet_SharedSupervised')

        acc1 = CalculateAccuracy(sess, mnist_test_x, mnist_label_test)
        fashionAcc = CalculateAccuracy(sess, fashion_test_x, fashion_test_label)
        svhnAcc = CalculateAccuracy(sess, svhn_test_x, svhn_label_test)
        cifarAcc = CalculateAccuracy(sess, cifar_test_x, cifar_test_label)

        print(acc1)
        print('\n')
        print(fashionAcc)
        print('\n')
        print(svhnAcc)
        print('\n')
        print(cifarAcc)
        print('\n')
        bc = 0
    else:
        # training

        #n_epochs = 10

        stateState = np.zeros((batch_size, 4))
        stateState[:, 0] = 1
        stateState[:, 1] = 1
        stateState[:, 2] = 1
        stateState[:, 3] = 1
        x_test = 0
        test_labels = 0

        for taskIndex in range(currentTask):

            if taskIndex == 0:
                x_train = mnist_train_x
                train_labels = mnist_train_label
                x_fixed = x_train[0:batch_size]
                x_test = mnist_test_x
                test_labels = mnist_label_test
            elif taskIndex == 1:
                x_train = fashion_train_x
                train_labels = fashion_train_label
                x_fixed = x_train[0:batch_size]
                x_test = fashion_test_x
                test_labels = fashion_test_label
            elif taskIndex == 2:
                x_train = svhn_train_x
                train_labels = svhn_label
                x_fixed = x_train[0:batch_size]
                x_test = svhn_test_x
                test_labels = svhn_label_test
            elif taskIndex == 3:
                x_train = cifar_train_x
                train_labels = cifar_train_label
                x_fixed = x_train[0:batch_size]
                x_test = cifar_test_x
                test_labels = cifar_test_label

            n_samples = np.shape(x_train)[0]
            total_batch = int(n_samples / batch_size)

            mnistAccArr = []
            fashionArr = []
            svhnArr = []
            cifarArr = []
            for epoch in range(n_epochs):
                # Random shuffling
                index = [i for i in range(np.shape(x_train)[0])]
                random.shuffle(index)
                x_train = x_train[index]
                train_labels = train_labels[index]

                # Loop over all batches
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    batch_xs_input = x_train[i * batch_size:i * batch_size + batch_size]
                    batch_xs_target = batch_xs_input
                    batch_labels = train_labels[i * batch_size:i * batch_size + batch_size]

                    if ADD_NOISE:
                        batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                        batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                    _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                        (train_op, loss, neg_marginal_likelihood, KL_divergence),
                        feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 1.0, task_state: stateState,y_in:batch_labels})

                    _, loss2 = sess.run(
                        (train_classifier, classLoss),
                        feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 1.0, task_state: stateState,
                                   y_in: batch_labels})

                    print("epoch %f: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                        epoch, tot_loss, loss_likelihood, loss_divergence))

                y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1,task_state:stateState,y_in:batch_labels})
                y_RPR = np.reshape(y_PRR, (-1, 32, 32,3))
                ims("results/" + "VAE" + str(epoch) + ".jpg", merge2(y_RPR[:64], [8, 8]))

                '''
                acc1 = CalculateAccuracy(sess, mnist_test_x, mnist_label_test)
                mnistAccArr.append(acc1)
    
                fashionAcc = CalculateAccuracy(sess, fashion_test_x, fashion_test_label)
                fashionArr.append(fashionAcc)
                svhnAcc = CalculateAccuracy(sess, svhn_test_x, svhn_label_test)
                svhnArr.append(svhnAcc)
                cifarAcc = CalculateAccuracy(sess, cifar_test_x, cifar_test_label)
                cifarArr.append(cifarAcc)
                '''
                if epoch > 0:
                    x_fixed_image = np.reshape(x_fixed, (-1, 32, 32,3))
                    ims("results/" + "Real" + str(epoch) + ".jpg", merge2(x_fixed_image[:64], [8, 8]))

            '''
            mnistAccArr = np.array(mnistAccArr).astype('str')
            myThirdName = "results/Mixture4_Lifelong_Conditional_task"+str(taskIndex)+".txt"
            f = open(myThirdName, "w", encoding="utf-8")
            for i in range(np.shape(mnistAccArr)[0]):
                f.writelines(mnistAccArr[i])
                f.writelines('\n')
            f.flush()
            f.close()
    
            if taskIndex > 0:
                fashionArr = np.array(fashionArr).astype('str')
                myThirdName = "results/Mixture4_Lifelong_Conditional_Fashion_task" + str(taskIndex) + ".txt"
                f = open(myThirdName, "w", encoding="utf-8")
                for i in range(np.shape(fashionArr)[0]):
                    f.writelines(fashionArr[i])
                    f.writelines('\n')
                f.flush()
                f.close()
            if taskIndex > 1:
                svhnArr = np.array(svhnArr).astype('str')
                myThirdName = "results/Mixture4_Lifelong_Conditional_SVHN_task" + str(taskIndex) + ".txt"
                f = open(myThirdName, "w", encoding="utf-8")
                for i in range(np.shape(svhnArr)[0]):
                    f.writelines(svhnArr[i])
                    f.writelines('\n')
                f.flush()
                f.close()
            if taskIndex > 2:
                cifarArr = np.array(cifarArr).astype('str')
                myThirdName = "results/Mixture4_Lifelong_Conditional_CIFAR_task" + str(taskIndex) + ".txt"
                f = open(myThirdName, "w", encoding="utf-8")
                for i in range(np.shape(cifarArr)[0]):
                    f.writelines(cifarArr[i])
                    f.writelines('\n')
                f.flush()
                f.close()
            '''

            #select the most relevant component
            score = sess.run(totalScore, feed_dict={x_hat: x_fixed, keep_prob: 1,task_state:stateState,y_in:batch_labels})
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

        saver.save(sess, 'models/LifelongMixture_32New_Dirchlet_SharedSupervised')
