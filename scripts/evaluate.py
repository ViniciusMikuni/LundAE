import argparse
import h5py
from math import *
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os, ast
import sys
from sklearn import metrics

#np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
#from MVA_cfg import *
import provider
import gapnet_seg as MODEL
#import  gapnet_classify_global as model


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--params', default='[50,1,32,64,128,128,2,64,128,128,256,256,256]', help='DNN parameters[[k,H,A,F,F,F,H,A,F,C,F]]')
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--model_path', default='../logs/PU/model.ckpt', help='Model checkpoint path')
parser.add_argument('--modeln', type=int,default=-1, help='Model number')
parser.add_argument('--batch', type=int, default=64, help='Batch Size  during training [default: 64]')
parser.add_argument('--num_point', type=int, default=500, help='Point Number [default: 500]')
parser.add_argument('--data_dir', default='../h5', help='directory with data [default: ../data/PU]')
parser.add_argument('--nfeat', type=int, default=8, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')
parser.add_argument('--dr',  default='0.01', help='deltaR matching [default: 0.01]')
parser.add_argument('--pt',  default='0.01', help='pT matching [default: 0.01]')
parser.add_argument('--is_data', dest='is_data', default=False, action='store_true')
parser.add_argument('--is_comp', dest='is_comp', default=False, action='store_true')

FLAGS = parser.parse_args()
MODEL_PATH = FLAGS.model_path
params = ast.literal_eval(FLAGS.params)
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)
H5_OUT = FLAGS.h5_folder
if not os.path.exists(H5_OUT): os.mkdir(H5_OUT)  

# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch
NFEATURES = FLAGS.nfeat


NUM_CATEGORIES = FLAGS.ncat
#Only used to get how many parts per category

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(NUM_POINT))
print('#### Using GPUs: {0}'.format(FLAGS.gpu))



    
print('### Starting evaluation')


EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'eval_files_gc_Dr{}_Pt{}.txt'.format(FLAGS.dr,FLAGS.pt)))
if FLAGS.is_data:
    EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'eval_files_data.txt'))
if FLAGS.is_comp:
    EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'eval_files_Dr004_Pt01.txt'))

#print("Loading: ",os.path.join(H5_DIR, 'eval_files_Dr{}_Pt{}.txt'.format(FLAGS.dr,FLAGS.pt)))

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl,truth_pl,  labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES)
          
            batch = tf.Variable(0, trainable=False)
                        
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,params=params,num_class=NUM_CATEGORIES)                        
            loss_CE = MODEL.get_loss_CE(pred, labels_pl)
            pred=tf.nn.softmax(pred)
            loss = loss_CE+ MODEL.get_loss_CD(tf.multiply(tf.reshape(pred[:,:,2],[BATCH_SIZE, NUM_POINT,1]),pointclouds_pl[:,:,:3]),truth_pl)
            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        if FLAGS.modeln >=0:
            saver.restore(sess,os.path.join(MODEL_PATH,'model_{}.ckpt'.format(FLAGS.modeln)))
        else:
            saver.restore(sess,os.path.join(MODEL_PATH,'model.ckpt'))
        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'truth_pl': truth_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,}
            
        eval_one_epoch(sess,ops)

def preprocessing(inputs,output):
    #Take the 3 momentum for each input and output sets
    inputs_changed = inputs[:,:,:3] #eta,phi,log(pt)    
    output_changed = output[:,:,:3]    
    #output_changed[:,:,2] = np.ma.log(output_changed[:,:,2]).filled(0)
    zeros = np.zeros((inputs_changed.shape))
    zeros[:,:output_changed.shape[1]]+=output_changed
    return zeros

def get_batch(data,label,truth, start_idx, end_idx):
    batch_label = label[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:,:]
    batch_truth = truth[start_idx:end_idx,:,:]
    return batch_data, batch_label,batch_truth

        
def eval_one_epoch(sess,ops):
    is_training = False

    total_correct = total_sig=total_correct_ones =  total_seen =total_seen_ones= loss_sum =0    
    eval_idxs = np.arange(0, len(EVALUATE_FILES))
    y_pred = []
    for fn in range(len(EVALUATE_FILES)):
        current_file = os.path.join(H5_DIR,EVALUATE_FILES[eval_idxs[fn]])
        current_data, current_label = provider.load_h5(current_file,'seg')
        full_data = current_data
        if current_data.shape[2]>NFEATURES:
            print('puppi not used')
            current_data = current_data[:,:,:NFEATURES]
        if current_data.shape[1]>NUM_POINT:
            print('Using less points')
            current_data = current_data[:,:NUM_POINT]
            current_label = current_label[:,:NUM_POINT]

        add_list = ['PFNoPU','puppiPU','chs','NPU','CHS_MET','PUPPI_MET',
                    #'puppiNoPU',
                ]        
        adds = provider.load_add(current_file,add_list)
        if not FLAGS.is_data:
            current_truth = adds['PFNoPU']
            current_truth = preprocessing(current_data,current_truth)
        else:
            add_list.append('nLeptons')
            current_truth = np.zeros((current_data.shape))

        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE        
        #num_batches = 1
        # if FLAGS.is_data:
        #     num_batches = 600

        for batch_idx in range(num_batches):
            scores = np.zeros(NUM_POINT)
            true = np.zeros(NUM_POINT)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data, batch_label,batch_truth = get_batch(current_data, current_label,current_truth, start_idx, end_idx)
            
            cur_batch_size = end_idx-start_idx


            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['truth_pl']: batch_truth,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training,
            }
            #,beforemax
            loss, pred = sess.run([ops['loss'], ops['pred']],feed_dict=feed_dict)         
            pred_val = np.argmax(pred, 2)

            correct_ones = pred_val*batch_label
            total_sig+=np.sum(batch_label==2)
            total_correct_ones +=np.sum(correct_ones==4)
                        
            loss_sum += np.mean(loss)
            if len(y_pred)==0:
                y_pred= pred[:,:,2]
                y_data = full_data[start_idx:end_idx]
                y_lab = batch_label
                y_add = {}
                for add in adds:
                    y_add[add] = adds[add][start_idx:end_idx]
            else:
                y_pred=np.concatenate((y_pred,pred[:,:,2]),axis=0)
                y_data=np.concatenate((y_data,full_data[start_idx:end_idx]),axis=0)
                y_lab=np.concatenate((y_lab,batch_label),axis=0)
                for add in adds:
                    y_add[add] = np.concatenate((y_add[add],adds[add][start_idx:end_idx]),axis=0)

    if not FLAGS.is_data:
        print('The signal accuracy is {0}'.format(total_correct_ones / float(total_sig)))
        flat_pred = y_pred.flatten()
        flat_lab = y_lab.flatten()
        flat_lab = flat_lab ==2
        results = metrics.roc_curve(flat_lab, flat_pred)
        threshs = [
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.90,
            0.95]
        with open(os.path.join(MODEL_PATH,'cut_eff.txt'),'w') as f:
            for thresh in threshs:
                bin = np.argmax(results[1]>thresh)
                cut = results[2][bin]            
                f.write('eff: {}, fpr: {}, cut: {} \n'.format(results[1][bin],results[0][bin],cut))
    with h5py.File(os.path.join(H5_OUT,'{0}.h5'.format(FLAGS.name)), "w") as fh5:
        dset = fh5.create_dataset("DNN", data=y_pred)
        dset = fh5.create_dataset("data", data=y_data)
        dset = fh5.create_dataset("pid", data=y_lab)
        for add in adds:
            dset = fh5.create_dataset(add, data=y_add[add])


################################################          
    

if __name__=='__main__':
  eval()
