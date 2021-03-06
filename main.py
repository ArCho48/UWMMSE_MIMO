from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import pdb
import time
import math
import pickle
import random
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

#from utils import *
from model import UWMMSE

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('datasetID', 'set1', 'Dataset string.')
flags.DEFINE_string('modelID', 'set1', 'Dataset string.')
flags.DEFINE_string('resultID', 'set1', 'Result string.')
flags.DEFINE_integer('min_users', 5, 'Minimum number of users in ad-hoc network.')
flags.DEFINE_integer('max_users', 50, 'Maximum number of users in ad-hoc network.')
flags.DEFINE_integer('tr_usr_steps', 2, 'Skip sizes in training.')
flags.DEFINE_float('pmax', 1.0, 'Maximum available power.')
flags.DEFINE_integer('tx_antennas', 1, 'Transmitter antennas.')
flags.DEFINE_integer('rx_antennas', 1, 'Receiver antennas.')
flags.DEFINE_integer('signal_dim', 1, 'Signal dimension.')
flags.DEFINE_integer('var_db', -114, 'Channel noise variance.')
flags.DEFINE_float('pl_exp', 2.2, 'Path loss exponent.')
flags.DEFINE_float('alpha', 1.0, 'Rayleigh dist scale.')
flags.DEFINE_string('expID', 'uwmmse', 'Experiment name.')
flags.DEFINE_string('mode', 'train', 'Experiment mode.')
flags.DEFINE_integer('hidden1', 5, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('unrolled_layers', 4, 'UWMMSE layers.')
flags.DEFINE_integer('wmmse_iters', 100, 'WMMSE iterations.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('tr_iter', 15000, 'Number of train iterations.')
flags.DEFINE_integer('val_smpls', 6400, 'Number of valid samples.')
flags.DEFINE_integer('te_smpls', 10048, 'Number of test samples.')

# Define Paths
dataPath = 'data/'+FLAGS.datasetID+'/'
modelPath = 'models/'+FLAGS.modelID+'/'
resultPath = 'results/'+FLAGS.resultID+'/'

# Noise power
var = 10**(FLAGS.var_db/10)

# Rician channel params
k_dB = 20
k = 10**(k_dB/10)
mu_ric = np.sqrt(k/(2*(k+1)))
sig_ric = np.sqrt(1/(2*(k+1)))

# Pickle Load
def pload( path ):
    dump = pickle.load( open( path, 'rb' ) )
    return( dump )
    
# Pickle dump
def pdump( dump, path ):
    f = open(path,'wb')
    pickle.dump(dump, f)
    f.close()

# Rayleigh channel
def generate_rayleigh_CSI(K, num_H, rng, diag_ratio=1):
    CH = 1 / np.sqrt(2) * (rng.randn(K, K,FLAGS.rx_antennas,FLAGS.tx_antennas) + 1j * rng.randn(K, K,FLAGS.rx_antennas,FLAGS.tx_antennas))

    return np.absolute(CH)

# Rician channel
def generate_rice_CSI(K, num_H, rng):
    CH = sig_ric * (mu_ric + rng.randn(K, K,FLAGS.rx_antennas,FLAGS.tx_antennas) + 1j * (mu_ric + rng.randn(K, K,FLAGS.rx_antennas,FLAGS.tx_antennas)))

    return np.absolute(CH)

# Geometric channel
def generate_geometry_CSI(K, num_H, rng, area_length=10, alpha=2):
    tx_pos = np.zeros([num_H, K, 2])
    rx_pos = np.zeros([num_H, K, 2])
    rayleigh_coeff = np.zeros([num_H, K, K, FLAGS.rx_antennas,FLAGS.tx_antennas])
    for i in range(num_H):
        tx_pos[i, :, :] = rng.rand(K, 2) * area_length
        rx_pos[i, :, :] = rng.rand(K, 2) * area_length
        rayleigh_coeff[i, :, :] = (
            np.square(rng.randn(K, K,FLAGS.rx_antennas,FLAGS.tx_antennas)) + np.square(rng.randn(K, K,FLAGS.rx_antennas,FLAGS.tx_antennas))) / 2

    tx_pos_x = np.reshape(tx_pos[:, :, 0], [num_H, K, 1]) + np.zeros([1, 1, K])
    tx_pos_y = np.reshape(tx_pos[:, :, 1], [num_H, K, 1]) + np.zeros([1, 1, K])
    rx_pos_x = np.reshape(rx_pos[:, :, 0], [num_H, 1, K]) + np.zeros([1, K, 1])
    rx_pos_y = np.reshape(rx_pos[:, :, 1], [num_H, 1, K]) + np.zeros([1, K, 1])
    d = np.sqrt(np.square(tx_pos_x - rx_pos_x) +
                np.square(tx_pos_y - rx_pos_y))
    G = np.divide(1, 1 + d**alpha)
    G = np.expand_dims(np.expand_dims(G,-1),-1) * rayleigh_coeff
    
    return np.squeeze(np.sqrt(G))

# Generate Test data
def genTeData(usrs):
    tS = []
    tH = []
    for i in range(int(FLAGS.te_smpls/FLAGS.batch_size)):
        K = random.sample(usrs,1)[0]
        tS.append(K)
        rng = np.random.RandomState(np.random.randint(1000,5000))
        for j in range(FLAGS.batch_size):
            #CH = generate_rayleigh_CSI(K, 1, rng, diag_ratio=1)
            #CH = generate_rice_CSI(K, 1, rng)
            CH = generate_geometry_CSI(K, 1, rng, area_length=K, alpha=3)
            tH.append(CH)
        
    return(tS,tH)

# Generate Train data
def genTrData(K):
    H = []
    rng = np.random.RandomState(np.random.randint(1000,5000))
    for i in range(FLAGS.batch_size):
        #CH = generate_rayleigh_CSI(K, 1, rng, diag_ratio=1)
        #CH = generate_rice_CSI(K, 1, rng)        
        CH = generate_geometry_CSI(K, 1, rng, area_length=K, alpha=3)
        H.append(CH)
       
    return( np.asarray(H) )

# Post process result
def proc_res(sum_rate,test_sizes):
    sizes = np.unique(test_sizes)
    msrs = []
    mean_sr = np.mean(sum_rate,axis=1)
    for i in range(sizes.shape[0]):
        msrs.append( np.mean(mean_sr[test_sizes==sizes[i]]) )
        
    return(msrs,sizes)
    
# Number of variables
def num_var():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    
    return(total_parameters)

# Create Model Instance
def create_model( session, exp='uwmmse', layers=4 ):
    # Create
    model = UWMMSE( Pmax=FLAGS.pmax, tx_antennas=FLAGS.tx_antennas, rx_antennas=FLAGS.rx_antennas, signal_dim=FLAGS.signal_dim, var=var, batch_size=FLAGS.batch_size, layers=layers, learning_rate=FLAGS.learning_rate, exp=exp )
    
    if exp == 'uwmmse':
        print("UWMMSE model created with {} trainable parameters\n".format(num_var()))
    
    # Initialize variables ( To train from scratch )
    session.run(tf.compat.v1.global_variables_initializer())
    
    return model

# Train
def mainTrain():
    # Create/Load dataset
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
        usrs = list(range(FLAGS.min_users,FLAGS.max_users+1,1))
        test_sizes, test_H = genTeData(usrs)
        pdump( test_sizes, dataPath+'test_sizes.pkl' )
        pdump( test_H, dataPath+'test_H.pkl' )
        print("Created dataset")
    else:
        test_sizes = pload( dataPath+'test_sizes.pkl' )
        test_H = pload( dataPath+'test_H.pkl' )
        print("Loaded dataset")
        
    # Initiate TF session for WMMSE
    with tf.compat.v1.Session(config=config) as sess:
        # WMMSE experiment
        if FLAGS.expID == 'wmmse':
        
            # Create model 
            model = create_model( sess, FLAGS.expID, FLAGS.wmmse_iters )
            
            # Test
            test_iter = FLAGS.te_smpls
                    
            print( '\nWMMSE Started\n' )

            t = 0.
            test_rate = 0.0
            sum_rate = []
            power = []

            for batch in range(0,test_iter,FLAGS.batch_size):
                batch_test_inputs = test_H[batch:batch+FLAGS.batch_size]
                start = time.time()
                avg_rate, batch_rate, batch_power = model.eval( sess, inputs=batch_test_inputs)
                t += (time.time() - start)
                test_rate += -avg_rate
                sum_rate.append( batch_rate )
                power.append(batch_power)
            
            test_rate /= test_iter
            test_rate *= FLAGS.batch_size

            # Average per-iteration test time
            t = t / test_iter
            t *= FLAGS.batch_size
            
            log = "Test_rate = {:.3f}, Time = {:.3f} sec\n"
            print(log.format( test_rate, t))    
            
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            
            if np.unique(test_sizes).shape[0] > 1:
                mean_sum_rate, sizes = proc_res(sum_rate, test_sizes)
                pdump( mean_sum_rate, resultPath+'wmmse_rate.pkl' )
                pdump( sizes, resultPath+'sizes.pkl' )
            else:
                mean_sum_rate = np.mean( sum_rate, axis=1 )
                sum_rate = np.concatenate( sum_rate, axis=0 )
                pdump( sum_rate, resultPath+'wmmse_rate.pkl' )
                
            pdump( power, resultPath+'wmmse_power.pkl' )
        else:
            # Create model 
            model = create_model( sess, FLAGS.expID, FLAGS.unrolled_layers )

            if FLAGS.mode == 'train':
                # Create model path
                if not os.path.exists(modelPath):
                    os.makedirs(modelPath)
        
                max_rate = 0.
                train_iter = FLAGS.tr_iter
                train_H = []
                
                usrs = list(range(FLAGS.min_users,FLAGS.max_users+1,FLAGS.tr_usr_steps))

                #Training loop
                print( '\nUWMMSE Training Started\n' )
                
                start = time.time()
                train_rate = 0.0
                
                for it in range(train_iter):
                    usr = random.sample(usrs,1)[0]
                    batch_train_inputs = genTrData(usr)
                    step_rate, batch_rate, power  = model.train( sess, inputs=batch_train_inputs) 
                    if np.isnan(step_rate) or np.isinf(step_rate) :
                        pdb.set_trace()
                    
                    train_rate += -step_rate
                
                    if ( ( (it+1) % 500 ) == 0):
                        train_rate /= 500
                                            
                        # Validate
                        val_rate = 0.0
                        val_iter = FLAGS.val_smpls
                        
                        for batch in range(val_iter):
                            usr = random.sample(usrs,1)[0]
                            batch_val_inputs = genTrData(usr)
                            avg_rate, batch_rate, batch_power = model.eval( sess, inputs=batch_val_inputs)
                            if np.isnan(avg_rate) or np.isinf(avg_rate):
                                pdb.set_trace()
                            val_rate += -avg_rate
                            
                        val_rate /= val_iter

                        log = "Iters {}/{}, Train Sum_rate = {:.3f}, \nValid Sum_rate = {:.3f}, Time Elapsed = {:.3f} sec\n"
                        print(log.format( it+1, FLAGS.tr_iter, train_rate, val_rate, time.time() - start) )
                                            
                        train_rate = 0.0

                        if (val_rate > max_rate):
                            max_rate = val_rate
                            model.save(sess, path=modelPath+'uwmmse-model', global_step=(it+1))

                print( 'Training Complete' )
            
            # Test
            t = 0.
            test_rate = 0.0
            test_iter = FLAGS.te_smpls
            
            power = []
            sum_rate = []
            
            # Restore best saved model
            model.restore(sess, path=modelPath)

            print( '\nUWMMSE Testing Started\n' )

            for batch in range(0,test_iter,FLAGS.batch_size):
                batch_test_inputs = test_H[batch:batch+FLAGS.batch_size]
                start = time.time()
                avg_sum_rate, batch_rate, batch_power = model.eval( sess, inputs=batch_test_inputs)
                t += (time.time() - start)
                if np.isnan(avg_sum_rate) or np.isinf(avg_sum_rate):
                    pdb.set_trace()
                sum_rate.append( batch_rate )
                power.append( batch_power )
                test_rate += (-avg_sum_rate * batch_test_inputs[0].shape[1])
            
            test_rate /= test_iter
            test_rate *= FLAGS.batch_size
            
            ## Average per-iteration test time   
            t = t / test_iter
            t *= FLAGS.batch_size
            
            log = "Test Sum_rate = {:.3f}, Time = {:.3f} sec\n"
            print(log.format( test_rate, t))
            
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            
            if np.unique(test_sizes).shape[0] > 1:
                mean_sum_rates, sizes = proc_res(sum_rate,test_sizes)
                pdump( mean_sum_rates, resultPath+'uwmmse_rate.pkl' )
                pdump( sizes, resultPath+'sizes.pkl' )
            else:
                mean_sum_rate = np.mean( sum_rate, axis=1 )
                sum_rate = np.concatenate( sum_rate, axis=0 )
                pdump( sum_rate, resultPath+'uwmmse_rate.pkl' )
                
            pdump( power, resultPath+'uwmmse_power.pkl' )

if __name__ == "__main__":        
    import sys

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    tf.compat.v1.set_random_seed(rn)
    np.random.seed(rn1)

    mainTrain()
