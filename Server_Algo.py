#coding:utf-8
import sys
import h5py
import cv2 as cv
import random
import time
import os
sys.path.append('/home/sal/caffe/python')
import caffe
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from progressbar import ETA, Bar, Percentage, ProgressBar


model_path = './models'


def SetUp(model_path):
    #Loading K-Means Model
    try:
        k_means_model = joblib.load('./models/k_means_model.pkl')
    except:
        print "No pretrained K-Means model was found! This is not an error"


    model_def = get_filename(model_path, 'prototxt')
    model_weights = get_filename(model_path, 'caffemodel')

    caffe.set_device(0)
    caffe.set_mode_cpu()
    try:
        net = caffe.Net(model_def, model_weights, caffe.TEST)
    except:
        print "No pretrained caffe model was found! Aborting...Please make sure that you put the caffe model in right directory and try again!"

    return net,k_means_model


def DragFeature(img_list,net,batch_size,W,H,fea_dim,name):

    accumulate_num =0
    batch_array = np.zeros((batch_size,3,H,W), dtype= np.float32)
    draged_features = np.zeros((len(img_list),fea_dim), dtype= np.float64)



    widgets = ["processing with batch_size: #%d|" % batch_size, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=len(img_list), widgets=widgets)
    pbar.start()

    for index in xrange(len(img_list)):
        pbar.update(index)

        img = cv.imread(img_list[index])
        img = cv.resize(img,(W,H))
        img = np.transpose(img,(2,0,1))

        if accumulate_num< batch_size:
            batch_array[accumulate_num,:,:,:] = img
            accumulate_num += 1
        if accumulate_num == batch_size:
            net.blobs['data'].data[:,:,:,:] = batch_array
            output = net.forward()
            try:
                feature = net.blobs['fc_out'].data[:]
            except:
                print "Feature can not be processed: %d"%(index)
            draged_features[len(img_list) - batch_size+1 : index+1, :] = feature
            accumulate_num = 0

    if accumulate_num:
        net.blobs['data'].data[:, :, :, :] = batch_array
        output = net.forward()
        try:
            feature = net.blobs['fc_out'].data[:]
        except:
            print "Feature can not be processed: %d" % (index)
        draged_features[len(img_list) - accumulate_num :, :] = feature[:accumulate_num,:]
        accumulate_num = 0

    with h5py.File('./features/'+name+'.h5', 'w') as feature_save:
        feature_save.create_dataset('feature', data= draged_features)

    return draged_features






def get_filename(path,suffix):
    f_list = os.listdir(path)
    for i in f_list:
        if i.endswith(suffix):
            return os.path.join(path,i)



def K_Means_train(feature_bank, cluster_num = -1, num_show = 100):
    if(cluster_num>0):
        print "K-Means Clustering..."
        k_means_model = KMeans(n_clusters=cluster_num,max_iter=100000,tol=1e-4)
        s = k_means_model.fit(feature_bank)
        print 'Cluster Finish,Centers:'
        #print s
        print k_means_model.cluster_centers_
        print 'Labels for each Sample'
        print np.random.choice(k_means_model.labels_, num_show)
        joblib.dump(k_means_model, './models/k_means_model.pkl')#+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    return k_means_model


def K_Medoid(feature_bank, cluster_num = -1):

    return

def Cluster(feature_bank, cluster_num = -1):
    if(cluster_num>0):
        a =1
    return

def Load_pics(data_path):
    img_paths = []
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith('jpg') or f.endswith('bmp') or f.endswith('png'):
                img_paths.append(os.path.join(root,f))
    return img_paths




print "hehe"

