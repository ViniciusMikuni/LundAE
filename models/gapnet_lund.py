import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util
from gat_layers import attn_feature



def placeholder_inputs(batch_size, num_point, num_features):
    pointclouds_pf = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    adj_matrix = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_point))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    zero_mask = tf.placeholder(tf.float32, shape=(batch_size,num_point))
    return pointclouds_pf,adj_matrix,zero_mask,  labels_pl



def gap_block(k,n_heads,nn_idx,net,edge_size,bn_decay,weight_decay,is_training,scname,bn=True):
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, self_attention, locals = attn_feature(net, edge_size[1], nn_idx, activation=tf.nn.relu,
                                                            in_dropout=0.6,
                                                            coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,bn=bn,
                                                            layer=scname+'layer', k=k, i=i)
        attns.append(edge_feature)# This is the edge feature * att. coeff. activated by RELU, one per particle
        local_features.append(locals) #Those are the yij


    neighbors_features = tf.concat(attns, axis=-1)
    batch_size = net.get_shape().as_list()[0]
    net = tf.squeeze(net)
    if batch_size == 1:
        net = tf.expand_dims(net, 0)

    
    neighbors_features = tf.concat([tf.expand_dims(net, -2), neighbors_features], axis=-1)
    locals_transform = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)

    return neighbors_features, locals_transform




def get_generator(point_cloud, adj_matrix,zero_mask,
                  is_training, use_adj=False,
                  weight_decay=None, bn_decay=None,bn=True,scname=''):
    ''' input: BxNxF
    Use https://arxiv.org/pdf/1902.08570 as baseline
    output:BxNx(cats*segms)  '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    k=2
    if use_adj:
        adj = -adj_matrix
    else:
        adj = tf_util.pairwise_distance(point_cloud[:,:,4:],zero_mask) #lund plane coordinates
        
    n_heads = 1
    nn_idx = tf_util.knn(adj, k=k)
    
    net, locals_transform= gap_block(k,n_heads,nn_idx,point_cloud,('filter0',32),bn_decay,weight_decay,is_training,scname+'0',bn=bn)


    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=bn, is_training=is_training, scope=scname+'gapnet01', bn_decay=bn_decay)
    net01 = net


    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=bn, is_training=is_training, scope=scname+'gapnet02', bn_decay=bn_decay)
    
    net02 = net

    if use_adj:
        adj = -adj_matrix
    else:
        adj = tf_util.pairwise_distance(net,zero_mask)
        
    nn_idx = tf_util.knn(adj, k=k)    
    adj_conv = nn_idx
    n_heads = 1

    net, locals_transform1= gap_block(k,n_heads,nn_idx,net,('filter1',64),bn_decay,weight_decay,is_training,scname+'1',bn=bn)

    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=bn, is_training=is_training, scope=scname+'gapnet11', bn_decay=bn_decay)
    net11 = net


    #net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
    #                     bn=bn, is_training=is_training, scope=scname+'gapnet12', bn_decay=bn_decay)
    #net12 = net

    
    net = tf.concat([
        net01,
        net02,
        net11,
        locals_transform,
        locals_transform1
    ], axis=-1)


    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1], 
                         activation_fn=tf.nn.relu,
                         bn=bn, is_training=is_training, scope=scname+'agg', bn_decay=bn_decay)
    

    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope=scname+'maxpool')
    net = tf.reshape(net, [batch_size, -1]) 

    return net

def get_extractor1(net1, is_training, num_class,
                   weight_decay=None, bn_decay=None,bn=True,scname='',reversal=False):
    if reversal:
        net = flip_gradient(net1,1.0)
    else:
        net=net1
    net = tf_util.fully_connected(net, 128, bn=bn, is_training=is_training, activation_fn=tf.nn.relu,
                                  scope=scname+'fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope=scname+'dp1')
    net = tf_util.fully_connected(net, num_class,activation_fn=None, scope=scname+'fc4')

        
    return net


def get_extractor2(net2, is_training, num_class,params, 
                  weight_decay=None, bn_decay=None,bn=True,scname='',reversal=False):

    if reversal:
        net = flip_gradient(net2,40.0)
    else:
        net = net2
    net = tf_util.fully_connected(net, params[10], bn=bn, is_training=is_training, activation_fn=tf.nn.relu,
                                  scope=scname+'fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                           scope=scname+'dp1')

    net = tf_util.fully_connected(net, params[10], bn=bn, is_training=is_training, activation_fn=tf.nn.relu,
                                  scope=scname+'fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope=scname+'dp2')
    net = tf_util.fully_connected(net, params[11], bn=bn, is_training=is_training, activation_fn=tf.nn.relu,
                                  scope=scname+'fc3', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_class,activation_fn=None, scope=scname+'fc4')
        
    return net




def get_loss(pred, label,num_class,norms=None):
  """ pred: B,NUM_CLASSES
      label: B, """
  labels = tf.one_hot(indices=label, depth=num_class)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred)
  classify_loss = tf.reduce_mean(loss)        
  return classify_loss
 

