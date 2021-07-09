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

os.environ['CUDA_VISIBLE_DEVICES']='7'

distributions = tf.distributions
from Mixture_Models import *
import keras.datasets.cifar10 as cifar10

def file_name2_(file_dir):
    t1 = []
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "C:/CommonData//rendered_chairs/" + a1 + "/renders/*.png"
            img_path = glob(b1)
            t1.append(img_path)

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return  cc

def file_name_(file_dir):
    t1 = []
    file_dir = "E:/LifelongMixtureModel/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "E:/LifelongMixtureModel/data/images_background/" + a1 + "/renders/*.png"
            b1 = "E:/LifelongMixtureModel/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)


    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc


def file_name(file_dir):
    t1 = []
    file_dir = "../images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "../images_background/" + a1 + "/renders/*.png"
            b1 = "../images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc

def file_name2(file_dir):
    t1 = []
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "../rendered_chairs/" + a1 + "/renders/*.png"
            img_path = glob(b1)
            t1.append(img_path)

        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return  cc


# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, task_state, disentangledCount):
    # encoding
    mu1, sigma1 = Encoder_64(x_hat, "encoder1")
    mu2, sigma2 = Encoder_64(x_hat, "encoder2")
    mu3, sigma3 = Encoder_64(x_hat, "encoder3")
    mu4, sigma4 = Encoder_64(x_hat, "encoder4")

    z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1), 0, 1, dtype=tf.float32)
    z2 = mu2 + sigma2 * tf.random_normal(tf.shape(mu2), 0, 1, dtype=tf.float32)
    z3 = mu3 + sigma3 * tf.random_normal(tf.shape(mu3), 0, 1, dtype=tf.float32)
    z4 = mu4 + sigma4 * tf.random_normal(tf.shape(mu4), 0, 1, dtype=tf.float32)

    s1 = Generator_64(z1, "decoder1")
    s2 = Generator_64(z2, "decoder2")
    s3 = Generator_64(z3, "decoder3")
    s4 = Generator_64(z4, "decoder4")

    imageSize = 64
    s1_1 = tf.reshape(s1,(-1,imageSize*imageSize*3))*task_state[:, 0:1]
    s2_1 = tf.reshape(s2,(-1,imageSize*imageSize*3))*task_state[:, 1:2]
    s3_1 = tf.reshape(s3,(-1,imageSize*imageSize*3))*task_state[:, 2:3]
    s4_1 = tf.reshape(s4,(-1,imageSize*imageSize*3))*task_state[:, 3:4]

    reco = s1_1 + s2_1 + s3_1 + s4_1
    reco = reco / (task_state[0, 0] + task_state[0, 1] + task_state[0, 2] + task_state[0, 3])
    reco = tf.reshape(reco,(-1,imageSize,imageSize,3))

    #Calculate task relationship

    # Select tasks
    reco1 = tf.reduce_mean(tf.reduce_sum(tf.square(s1 - x_hat), [1, 2, 3]))
    reco2 = tf.reduce_mean(tf.reduce_sum(tf.square(s2 - x_hat), [1, 2, 3]))
    reco3 = tf.reduce_mean(tf.reduce_sum(tf.square(s3 - x_hat), [1, 2, 3]))
    reco4 = tf.reduce_mean(tf.reduce_sum(tf.square(s4 - x_hat), [1, 2, 3]))
    reco1_ = reco1 + (1 - task_state[0, 0]) * 1000000
    reco2_ = reco2 + (1 - task_state[0, 1]) * 1000000
    reco3_ = reco3 + (1 - task_state[0, 2]) * 1000000
    reco4_ = reco4 + (1 - task_state[0, 3]) * 1000000

    totalScore = tf.stack((reco1_, reco2_, reco3_, reco4_), axis=0)

    mixParameter = task_state[0]
    sum = mixParameter[0] + mixParameter[1] + mixParameter[2] + mixParameter[3]
    mixParameter = mixParameter / sum

    dist = tf.distributions.Dirichlet(mixParameter)
    mix_samples = dist.sample()

    b1 = mix_samples[0] * task_state[0, 0]
    b2 = mix_samples[1] * task_state[0, 1]
    b3 = mix_samples[2] * task_state[0, 2]
    b4 = mix_samples[3] * task_state[0, 3]

    mix_samples2 = tf.stack((b1,b2,b3,b4),axis=0)

    # loss
    reco1_loss = reco1 * mix_samples2[0]
    reco2_loss = reco2 * mix_samples2[1]
    reco3_loss = reco3 * mix_samples2[2]
    reco4_loss = reco4 * mix_samples2[3]

    # loss
    marginal_likelihood = (reco1_loss + reco2_loss + reco3_loss + reco4_loss)

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

    KL_divergence = k1 * mix_samples2[0] + k2 * mix_samples2[1] + k3 * mix_samples2[2] + k4 * mix_samples2[3]
    KL_divergence = KL_divergence

    p2 = 1

    gamma = 4
    loss = marginal_likelihood + gamma * tf.abs(KL_divergence - disentangledCount)

    z = z1
    y = reco

    return y, z, loss, -marginal_likelihood, KL_divergence,totalScore

def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y


n_hidden = 500
IMAGE_SIZE_MNIST = 28
dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image

dim_z = 256

# train
n_epochs = 100
batch_size = 64
learn_rate = 0.001

train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
n_samples = train_size
# input placeholders

# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[64, 64, 64, 3], name='input_img')
x = tf.placeholder(tf.float32, shape=[64, 64, 64, 3], name='target_img')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
task_state = tf.placeholder(tf.float32, shape=[64, 4])
disentangledCount = tf.placeholder(tf.float32)

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence,totalScore = autoencoder(x_hat, x, dim_img, dim_z,
                                                                 n_hidden, keep_prob,
                                                                 task_state, disentangledCount)

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

img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
data_files = img_path
data_files = sorted(data_files)
data_files = np.array(data_files)  # for tl.iterate.minibatches
celebaFiles = data_files

# load 3D chairs
img_path = glob('../CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
data_files = img_path
data_files = sorted(data_files)
data_files = np.array(data_files)  # for tl.iterate.minibatches
cacdFiles = data_files


file_dir = "../rendered_chairs/"
files = file_name2(file_dir)
data_files = files
data_files = sorted(data_files)
data_files = np.array(data_files)  # for tl.iterate.minibatches
chairFiles = data_files


files = file_name(1)
data_files = files
data_files = sorted(data_files)
data_files = np.array(data_files)  # for tl.iterate.minibatches
zimuFiles = data_files


saver = tf.train.Saver()

isWeight = False
currentTask = 4

def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str

isWeight = False
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

    if isWeight:
        saver.restore(sess, 'models/LifelongMixture_64_Dirichlet')

        img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches

        myIndex = 10
        celebaFiles = data_files[myIndex * batch_size:(myIndex + 2) * batch_size]

        # load 3D chairs
        img_path = glob('C:/CommonData/CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        cacdFiles = data_files[myIndex * batch_size:(myIndex + 2) * batch_size]

        file_dir = "C:/CommonData/rendered_chairs/"
        files = file_name2_(file_dir)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        chairFiles = data_files[myIndex * batch_size:(myIndex + 2) * batch_size]

        files = file_name_(1)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        zimuFiles = data_files[myIndex * batch_size:(myIndex + 2) * batch_size]

        dataArray = []
        for taskIndex in range(4):

            taskIndex = 2
            if taskIndex == 0:
                x_train = celebaFiles
                x_fixed = x_train[0:batch_size]
                x_fixed2 = x_train[batch_size:batch_size * 2]
            elif taskIndex == 1:
                x_train = cacdFiles
                x_fixed = x_train[0:batch_size]
                x_fixed2 = x_train[batch_size:batch_size * 2]
            elif taskIndex == 2:
                x_train = chairFiles
                x_fixed = x_train[0:batch_size]
                x_fixed2 = x_train[batch_size:batch_size * 2]
            elif taskIndex == 3:
                x_train = zimuFiles
                x_fixed = x_train[0:batch_size]
                x_fixed2 = x_train[batch_size:batch_size * 2]

            batchFiles = x_fixed
            batchFiles2 = x_fixed2

            if taskIndex == 0:
                batch = [get_image(
                    sample_file,
                    input_height=128,
                    input_width=128,
                    resize_height=64,
                    resize_width=64,
                    crop=True)
                    for sample_file in batchFiles]

                batch2 = [get_image(
                    sample_file,
                    input_height=128,
                    input_width=128,
                    resize_height=64,
                    resize_width=64,
                    crop=True)
                    for sample_file in batchFiles2]

            elif taskIndex == 1:
                batch = [get_image(
                    sample_file,
                    input_height=250,
                    input_width=250,
                    resize_height=64,
                    resize_width=64,
                    crop=True)
                    for sample_file in batchFiles]

                batch2 = [get_image(
                    sample_file,
                    input_height=250,
                    input_width=250,
                    resize_height=64,
                    resize_width=64,
                    crop=True)
                    for sample_file in batchFiles2]
            elif taskIndex == 2:
                image_size = 64
                batch = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                         for batch_file in batchFiles]
                batch2 = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                          for batch_file in batchFiles2]
            elif taskIndex == 3:
                batch = [get_image(batch_file, 105, 105,
                                   resize_height=64, resize_width=64,
                                   crop=False, grayscale=False) \
                         for batch_file in batchFiles]
                batch = np.array(batch)
                batch = np.reshape(batch, (64, 64, 64, 1))
                batch = np.concatenate((batch, batch, batch), axis=-1)

                batch2 = [get_image(batch_file, 105, 105,
                                    resize_height=64, resize_width=64,
                                    crop=False, grayscale=False) \
                          for batch_file in batchFiles2]
                batch2 = np.array(batch2)
                batch2 = np.reshape(batch2, (64, 64, 64, 1))
                batch2 = np.concatenate((batch2, batch2, batch2), axis=-1)

            dataArray.append(batch)

            x_fixed = batch
            x_fixed = np.array(x_fixed)

            x_fixed2 = batch2
            x_fixed2 = np.array(x_fixed2)

            # select the most relevant component
            stateState = np.zeros((batch_size, 4))
            stateState[:, 0] = 1
            stateState[:, 1] = 1
            stateState[:, 2] = 1
            stateState[:, 3] = 1
            score = sess.run(totalScore, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState})
            a = np.argmin(score, axis=0)
            index = a

            z = 0
            generator_outputs = 0
            if index == 0:
                mu1, sigma1 = Encoder_64(x_hat, "encoder1", reuse=True)
                z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1), 0, 1, dtype=tf.float32)
                Reco = Generator_64(z1, "decoder1", reuse=True)
                generator_outputs = Generator_64(z_in, "decoder1", reuse=True)
                z = z1
            elif index == 1:
                mu2, sigma2 = Encoder_64(x_hat, "encoder2", reuse=True)
                z2 = mu2 + sigma2 * tf.random_normal(tf.shape(mu2), 0, 1, dtype=tf.float32)
                Reco = Generator_64(z2, "decoder2", reuse=True)
                generator_outputs = Generator_64(z_in, "decoder2", reuse=True)
                z = z2
            elif index == 2:
                mu3, sigma3 = Encoder_64(x_hat, "encoder3", reuse=True)
                z3 = mu3 + sigma3 * tf.random_normal(tf.shape(mu3), 0, 1, dtype=tf.float32)
                Reco = Generator_64(z3, "decoder3", reuse=True)
                generator_outputs = Generator_64(z_in, "decoder3", reuse=True)
                z = z3
            elif index == 3:
                mu4, sigma4 = Encoder_64(x_hat, "encoder4", reuse=True)
                z4 = mu4 + sigma4 * tf.random_normal(tf.shape(mu4), 0, 1, dtype=tf.float32)
                Reco = Generator_64(z4, "decoder4", reuse=True)
                generator_outputs = Generator_64(z_in, "decoder4", reuse=True)
                z = z4

            code1 = sess.run(z, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState})
            code2 = sess.run(z, feed_dict={x_hat: x_fixed2, keep_prob: 1, task_state: stateState})


            recoArr = []
            minV = -3
            maxV = 3
            tv = 6.0 / 12.0

            '''
            for j in range(256):
                code1 = sess.run(z, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState})
                recoArr = []
                myIndex = 0
                for i in range(12):
                    code1[:, j] = minV + tv * i
                    myReco = sess.run(generator_outputs, feed_dict={z_in: code1, keep_prob: 1, task_state: stateState})
                    recoArr.append(myReco[myIndex])
                recoArr = np.array(recoArr)
                ims("results/" + "inter" + str(j) + ".png", merge2(recoArr, [1, 12]))
                bc = 2

            BC =0
            '''

            for t1 in range(64):
                for j in range(256):
                    code1 = sess.run(z, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState})
                    recoArr = []
                    j = 224
                    myIndex = t1
                    for i in range(12):
                        code1[:,j] = minV + tv * i
                        myReco = sess.run(generator_outputs, feed_dict={z_in: code1, keep_prob: 1, task_state: stateState})
                        recoArr.append(myReco[myIndex])
                    recoArr = np.array(recoArr)
                    ims("results/" + "inter" + str(t1) + ".png", merge2(recoArr, [1, 12]))
                    bc = 2
                    break

            c=0
            for t in range(2):
                if t ==1 :
                    t = t+20
                recoArr.append(x_fixed2[t])
                for i in range(10):
                    newCode = code2 + distance*i
                    myReco = sess.run(generator_outputs, feed_dict={z_in: newCode, keep_prob: 1, task_state: stateState})
                    recoArr.append(myReco[t])
                recoArr.append(x_fixed[t])

            recoArr = np.array(recoArr)

            ims("results/" + "inter" + str(taskIndex) + ".png", merge2(recoArr, [2, 12]))

            myReco = sess.run(Reco, feed_dict={x_hat: x_fixed, keep_prob: 1, task_state: stateState})
            ims("results/" + "Dataset" + str(taskIndex) + "_mini.png", merge2(x_fixed[:16], [2, 8]))
            ims("results/" + "Reco" + str(taskIndex) + "_H_mini.png", merge2(myReco[:16], [2, 8]))

            bc = 0
        bc = 0


    # training
    n_epochs = 20

    stateState = np.zeros((batch_size, 4))
    stateState[:, 0] = 1
    stateState[:, 1] = 1
    stateState[:, 2] = 1
    stateState[:, 3] = 1
    disentangledScore = 0.5
    vChange = 25.0 / n_epochs

    for taskIndex in range(currentTask):
        taskIndex = 1
        if taskIndex == 0:
            x_train = celebaFiles
            x_fixed = x_train[0:batch_size]
        elif taskIndex == 1:
            x_train = cacdFiles
            x_fixed = x_train[0:batch_size]
        elif taskIndex == 2:
            x_train = chairFiles
            x_fixed = x_train[0:batch_size]
        elif taskIndex == 3:
            x_train = zimuFiles
            x_fixed = x_train[0:batch_size]

        disentangledScore = disentangledScore + vChange

        n_samples = np.shape(np.array(x_train))[0]
        total_batch = int(n_samples / batch_size)

        for epoch in range(n_epochs):
            # Random shuffling
            index = [i for i in range(np.shape(x_train)[0])]
            random.shuffle(index)
            x_train = x_train[index]
            image_size = 64

            # Loop over all batches
            for i in range(total_batch):

                batchFiles = x_train[i * batch_size:i * batch_size + batch_size]
                if taskIndex == 0:
                    batch = [get_image(
                        sample_file,
                        input_height=128,
                        input_width=128,
                        resize_height=64,
                        resize_width=64,
                        crop=True)
                        for sample_file in batchFiles]
                elif taskIndex == 1:
                    batch = [get_image(
                        sample_file,
                        input_height=250,
                        input_width=250,
                        resize_height=64,
                        resize_width=64,
                        crop=True)
                        for sample_file in batchFiles]
                elif taskIndex == 2:
                    batch = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                             for batch_file in batchFiles]
                elif taskIndex == 3:
                    batch = [get_image(batch_file, 105, 105,
                                       resize_height=64, resize_width=64,
                                       crop=False, grayscale=False) \
                             for batch_file in batchFiles]
                    batch = np.array(batch)
                    batch = np.reshape(batch, (64, 64, 64, 1))
                    batch = np.concatenate((batch, batch, batch), axis=-1)

                # Compute the offset of the current minibatch in the data.
                batch_xs_target = batch
                x_fixed = batch
                batch_xs_input = batch

                if ADD_NOISE:
                    batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 1.0, task_state: stateState,disentangledCount:disentangledScore})

                print("epoch %f: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                    epoch, tot_loss, loss_likelihood, loss_divergence))

            y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1,task_state:stateState,disentangledCount:disentangledScore})
            y_RPR = np.reshape(y_PRR, (-1, 64, 64,3))
            ims("results/" + "VAE" + str(epoch) + ".jpg", merge2(y_RPR[:64], [8, 8]))

            if epoch > 0:
                x_fixed_image = np.reshape(x_fixed, (-1, 64, 64,3))
                ims("results/" + "Real" + str(epoch) + ".jpg", merge2(x_fixed_image[:64], [8, 8]))

        #select the most relevant component
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

        saver.save(sess, 'models/LifelongMixture_64_Dirichlet')
