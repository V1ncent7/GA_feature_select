import tensorflow as tf
#print tf.__version__
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from utils import *
import os
import argparse
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def dnn(GA_fea_dim):
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    Dtype = tf.float32
    NODE_FEATURE_DIM = GA_fea_dim
    EMBED_DIM = 64
    EMBED_DEPTH = 2
    OUTPUT_DIM = 64
    ITERATION_LEVEL = 5
    LEARNING_RATE = 1e-4
    MAX_EPOCH = 2
    BATCH_SIZE = 30
    LOAD_PATH = None
    LOG_PATH = None
    SAVE_PATH = './saved_model/graphnn-model'

    TEST_FREQ = 1
    SAVE_FREQ = 5

    #DATA_FILE_NAME = './data/acfgSSL_{}/'.format(NODE_FEATURE_DIM)
    DATA_FILE_NAME = 'D:\\GA\\result_json\\'
    #SOFTWARE=('openssl-1.0.1f-', 'openssl-1.0.1u-')
    #SOFTWARE = ('openssl-1.0.1f-',)
    OPTIMIZATION=('-O0', '-O1','-O2','-O3')
    COMPILER=('armeb-linux', )#'i586-linux', 'mips-linux')
    #COMPILER = ('armeb-linux',)
    VERSION=('v54',)
    '''

    DATA_FILE_NAME = 'E:\\test0602\\'
    # SOFTWARE = ('openssl-1.0.1f-',)  # 'openssl-1.0.1u-')
    OPTIMIZATION = ('-O0', '-O1', '-O2', '-O3',)
    # COMPILER = ('i586-linux', 'armeb-linux', 'mips-linux',)
    COMPILER = ('armeb-linux',)
    VERSION = ('v54',)
    '''


    # Process the input graphs

    F_NAME = get_f_name2(DATA_FILE_NAME, COMPILER, OPTIMIZATION, VERSION)

    #F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER,
    #        OPTIMIZATION, VERSION)

    FUNC_NAME_DICT = get_f_dict(F_NAME)
     

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    #print ("{} graphs, {} functions".format(len(Gs), len(classes)))


    if os.path.isfile('data2/class_perm.npy'):
        perm = np.load('data2/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('data2/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('data2/class_perm.npy', perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test =\
            partition_data(Gs,classes,[0.8,0.1,0.1],perm)
    '''
    print ("Train: {} graphs, {} functions".format(
            len(Gs_train), len(classes_train)))
    print ("Dev: {} graphs, {} functions".format(
            len(Gs_dev), len(classes_dev)))
    print ("Test: {} graphs, {} functions".format(
            len(Gs_test), len(classes_test)))
    '''
    # Fix the pairs for validation
    if os.path.isfile('data2/valid.json'):
        with open('data2/valid.json') as inf:
            valid_ids = json.load(inf)
        valid_epoch = generate_epoch_pair(
                Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    else:
        valid_epoch, valid_ids = generate_epoch_pair(
                Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        with open('data2/valid.json', 'w') as outf:
            json.dump(valid_ids, outf)

    # Model
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype,
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    # Train
    auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
            BATCH_SIZE, i = 0, load_data=valid_epoch)
    gnn.say("Initial training auc = {0} @ {1}".format(auc, datetime.now()))
    auc0, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
            BATCH_SIZE, i = 0, load_data=valid_epoch)
    gnn.say("Initial validation auc = {0} @ {1}".format(auc0, datetime.now()))

    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        l = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
        gnn.say("EPOCH {3}/{0}, loss = {1} @ {2}".format(
            MAX_EPOCH, l, datetime.now(), i))

        if (i % TEST_FREQ == 0):
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
                    BATCH_SIZE, i, load_data=valid_epoch)
            gnn.say("Testing model: training auc = {0} @ {1}".format(
                auc, datetime.now()))
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
                    BATCH_SIZE, 0, load_data=valid_epoch)
            gnn.say("Testing model: validation auc = {0} @ {1}".format(
                auc, datetime.now()))

            if auc > best_auc:
                path = gnn.save(SAVE_PATH+'_best')
                best_auc = auc
                gnn.say("Model saved in {}".format(path))

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))

    return best_auc
