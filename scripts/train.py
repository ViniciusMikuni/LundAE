import argparse
import math
import subprocess
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import socket
import os,ast
import sys
import time
from sklearn import metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
import provider
import gapnet_lund as MODEL


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=778, help='Point Number  [default: 100]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--wd', type=float, default=0.0, help='Decay rate for weight decay [default: 0.0]')
parser.add_argument('--decay_step', type=int, default=1000000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--data_dir', default='../h5', help='directory with data [default: hdf5_data]')
parser.add_argument('--nfeat', type=int, default=7, help='Number of features PF [default: 16]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--adj', action='store_true', default=False,help='Use adj matrix from file')

FLAGS = parser.parse_args()
DATA_DIR = FLAGS.data_dir
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FEAT = FLAGS.nfeat
NUM_CLASSES = FLAGS.ncat
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL_FILE = os.path.join(BASE_DIR, '..', 'models','gapnet_lund.py')
LOG_DIR = os.path.join('../logs',FLAGS.log_dir)

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LEARNING_RATE_CLIP = 1e-6
HOSTNAME = socket.gethostname()
EARLY_TOLERANCE=15


TRAIN_FILES = provider.getDataFiles(os.path.join(DATA_DIR, 'train_list.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(DATA_DIR, 'test_list.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl,adj_matrix,zero_mask,labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_FEAT)

            is_training = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            print("--- Get model and loss")

            with tf.variable_scope('TAGGER'):
                gen = MODEL.get_generator(pointclouds_pl,adj_matrix,zero_mask,
                                          is_training=is_training,use_adj = FLAGS.adj,
                                          scname='ENCODER',bn=True,
                                          bn_decay=bn_decay, weight_decay=FLAGS.wd)
                pred = MODEL.get_extractor1(gen, is_training=is_training,
                                            scname='DECODER',bn=True,
                                            bn_decay=bn_decay,num_class=NUM_CLASSES, 
                                            weight_decay=FLAGS.wd)
                   

            loss = MODEL.get_loss(pred, labels_pl,NUM_CLASSES)            
            pred = tf.nn.softmax(pred)

            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        
        
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        log_string("Total number of weights for the model: " + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        
        ops = {
            'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'adj_matrix':adj_matrix,            
            'zero_matrix':zero_mask,            
            'is_training': is_training,
            'pred': pred,


            'loss': loss,
            'train_op': train_op,
            'learning_rate':learning_rate,
            'merged': merged,
            'step': batch,
        }

        early_stop = np.inf
        earlytol = 0

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()            
            
            train_one_epoch(sess, ops, train_writer) 
            lss = eval_one_epoch(sess, ops, test_writer)
            cond = lss < early_stop 
            if cond:
                early_stop = lss
                earlytol = 0
                # Save the variables to disk.
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
                log_string("Model saved in file: %s" % save_path)
            else:            
                if earlytol >= EARLY_TOLERANCE:
                    break
                else:
                    print("No improvement for {0} epochs".format(earlytol))
                    earlytol+=1
            

def get_batch(data_pl,adj,label, start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_adj = adj[start_idx:end_idx]
    batch_data_pl = data_pl[start_idx:end_idx,:,:]
    return batch_data_pl,batch_adj, batch_label


def Preprocessing(data):
    npoints = data.shape[1]
    nfeatures = data.shape[2] - npoints
    new_data = data[:,:,:nfeatures]
    zero_mask = new_data[:,:,0] ==0
    new_data[:,:,0] = np.ma.log(new_data[:,:,0]).filled(0)
    new_data[:,:,1] = np.ma.log(new_data[:,:,1]).filled(0)
    new_data[:,:,2] = np.ma.log(new_data[:,:,2]).filled(0)
    new_data[:,:,3] = np.ma.log(new_data[:,:,3]).filled(0)
    adj_matrix = data[:,:,nfeatures:]

    return new_data,adj_matrix,zero_mask

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training=True
    
    loss_sum = 0
    log_string(str(datetime.now()))
    for TRAIN_FILE in TRAIN_FILES:
        current_data_pl, current_label = provider.load_lund(os.path.join(DATA_DIR,TRAIN_FILE))    
        current_data_pl, current_label, _ = provider.shuffle_data(current_data_pl, np.squeeze(current_label))
        current_data_pl,adj_matrix,zero_mask = Preprocessing(current_data_pl)

        file_size = current_data_pl.shape[0]
        num_batches = file_size // BATCH_SIZE
        #num_batches = 2
        for batch_idx in range(num_batches):
            
            start_idx = batch_idx * (BATCH_SIZE)
            end_idx = (batch_idx+1) * (BATCH_SIZE)
            batch_data_pl,batch_adj, batch_label = get_batch(current_data_pl,adj_matrix, current_label,start_idx, end_idx)
        
            feed_dict = {             
                ops['pointclouds_pl']: batch_data_pl,
                ops['labels_pl']: batch_label,
                ops['adj_matrix']: batch_adj,
                ops['zero_matrix']: zero_mask[start_idx:end_idx].astype(float), #for zero-padded
                ops['is_training']: is_training,
            }
        
            summary, step, _, loss = sess.run([ops['merged'], ops['step'],
                                               ops['train_op'],
                                               ops['loss']
                                           ],
                                              feed_dict=feed_dict)

    
            train_writer.add_summary(summary, step)
            loss_sum += loss

    log_string('mean loss: %f' % (loss_sum / float(num_batches*len(TRAIN_FILES))))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    loss_sum = 0
    y_source=[]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    for file_idx, TEST_FILE in enumerate(TEST_FILES):
        current_data_pl, current_label = provider.load_lund(os.path.join(DATA_DIR,TEST_FILE))
        current_data_pl, current_label, _ = provider.shuffle_data(current_data_pl, np.squeeze(current_label))
        current_data_pl,adj_matrix,zero_mask = Preprocessing(current_data_pl)
        file_size = current_data_pl.shape[0]
        num_batches = file_size // (BATCH_SIZE)
        #num_batches = 2
        
        
        for batch_idx in range(num_batches):
            
            start_idx = batch_idx * (BATCH_SIZE)
            end_idx = (batch_idx+1) * (BATCH_SIZE)
            batch_data_pl,batch_adj, batch_label = get_batch(current_data_pl,adj_matrix, current_label,start_idx, end_idx)
            
            feed_dict = {             
                ops['pointclouds_pl']: batch_data_pl,
                ops['labels_pl']: batch_label,
                ops['zero_matrix']: zero_mask[start_idx:end_idx].astype(float),
                ops['is_training']: is_training,
                ops['adj_matrix']: batch_adj,
            }
            
            if file_idx ==0 and batch_idx ==0:
                start_time = time.time()
            
            summary, step, loss,pred,lr = sess.run([ops['merged'], ops['step'],
                                                    ops['loss'],ops['pred'],
                                                    ops['learning_rate']
                                                ],
                                                   feed_dict=feed_dict)
        

            if file_idx ==0 and batch_idx ==0:
                duration = time.time() - start_time
                log_string("Eval time: "+str(duration)) 
                log_string("Learning rate: "+str(lr)) 



            test_writer.add_summary(summary, step)
           
            
            loss_sum += loss
            if len(y_source)==0:
                y_source = np.squeeze(pred)
                y_label = batch_label
            else:
                y_source=np.concatenate((y_source,np.squeeze(pred)),axis=0)
                y_label=np.concatenate((y_label,batch_label),axis=0)
            

    fpr, tpr, _ = metrics.roc_curve(y_label, y_source[:,1], pos_label=1)    
    log_string("AUC: {}".format(metrics.auc(fpr, tpr)))

    bineff = np.argmax(tpr>0.3)

    log_string('SOURCE: 1/effB at {0} effS = {1}'.format(tpr[bineff],1.0/fpr[bineff]))
    log_string('mean loss: %f' % (loss_sum*1.0 / float(num_batches*len(TEST_FILES))))
    EPOCH_CNT += 1


    return loss_sum*1.0 / float(num_batches)
    


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
