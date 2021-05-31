import pdb
import tensorflow.compat.v1 as tf

# UWMMSE
class UWMMSE(object):
        # Initialize
        def __init__( self, Pmax=1., tx_antennas=1, rx_antennas=1, signal_dim=1, var=7e-10, batch_size=64, layers=4, learning_rate=1e-3, max_gradient_norm=5.0, exp='uwmmse' ):
            self.Pmax              = tf.cast( Pmax, tf.float64 )
            self.variance          = var
            self.batch_size        = batch_size
            self.layers            = layers
            self.learning_rate     = learning_rate
            self.max_gradient_norm = max_gradient_norm
            self.exp               = exp
            self.global_step       = tf.Variable(0, trainable=False, name='global_step')
            self.T                 = tx_antennas
            self.R                 = rx_antennas
            self.d                 = signal_dim
            self.build_model()

        # Build Model
        def build_model(self):
            self.init_placeholders()
            self.build_network()
            self.build_objective()
            
        def init_placeholders(self):
            # CSI [Batch_size X Nodes X Nodes]
            self.H = tf.compat.v1.placeholder(tf.float64, shape=[None, None, None, self.R, self.T], name="H")
        
        # Normlize input to gcn
        def inp_norm(self):
            mu = tf.reduce_mean( self.H_gcn, axis=[-1], keepdims=True )
            sig = tf.math.reduce_std( self.H_gcn, axis=[-1], keepdims=True )
            
            H = tf.math.divide( ( self.H_gcn - mu ), sig )
            return(H)
            
        # Building network
        def build_network(self):
            # Retrieve number of nodes for initializing V
            self.nNodes = tf.shape( self.H )[1]
            
            # Extract diagonals
            self.dia = tf.transpose( tf.compat.v1.matrix_diag_part(tf.transpose(self.H,(0,3,4,1,2))), (0,3,1,2) )
            
            # Variance tensor
            self.var = self.variance * tf.eye( self.R, batch_shape=[self.nNodes] )
            self.var = tf.tile( tf.expand_dims( self.var, axis=0 ), tf.constant( [self.batch_size,1,1,1] ) )
            self.var = tf.cast( self.var, dtype=tf.float64 )

            # Maximum V = sqrt(Pmax)
            Vmax = tf.math.sqrt(self.Pmax/(self.T * self.d))

            if self.exp == 'uwmmse':
                # Learn GCN i/p
                w = tf.compat.v1.get_variable( name='w', shape=(self.R * self.T, 1), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                b = tf.compat.v1.get_variable( name='b', shape=(1,), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)

                if self.T < self.R:
                    self.H_gcn = tf.reshape( tf.transpose( self.H, (0,1,2,4,3) ), [ self.batch_size, self.nNodes, self.nNodes, self.R * self.T ] )
                else:
                    self.H_gcn = tf.reshape( self.H, [ self.batch_size, self.nNodes, self.nNodes, self.R * self.T ] )
                self.H_gcn = tf.add( tf.matmul( self.H_gcn, w ), b ) 
                self.H_gcn = tf.squeeze( self.H_gcn )
                self.H_gcn = self.inp_norm()
                
                # Diagonal for GCN i/p
                self.dH_gcn = tf.linalg.diag_part( self.H_gcn ) 
                self.dH_gcn = tf.compat.v1.matrix_diag( self.dH_gcn )
                        
            # Initialize V
            V = Vmax * tf.ones([self.batch_size, self.nNodes, self.T, self.d], dtype=tf.float64)#
            self.V = V

            # Iterate over layers l
            for l in range(self.layers):
                #with tf.compat.v1.variable_scope('Layer{}'.format(l+1)):
                with tf.variable_scope('Layer{}'.format(1), reuse=tf.AUTO_REUSE):
                    # Compute U^l
                    U = self.U_block( V )

                    # Compute W^l
                    W_wmmse = self.W_block( U, V )

                    if self.exp == 'wmmse':
                        # Compute V^l
                        V = self.V_block( U, W_wmmse )
                    else:
                        ## Learn a^l
                        a = self.gcn('a')
                        
                        ## Learn b^l
                        b = self.gcn('b')
                        
                        ## Compute Wcap^l = a^l * W^l + b^l
                        W = tf.math.add( tf.math.multiply( a, W_wmmse ), b )

                        # Compute V^l
                        V = self.V_block( U, W )
                    
                    ## Saturation non-linearity  ->  V if Tr(V'V) < Pmax ; V * sqrt(P)/Vfrobenius if Tr(V'V) > Pmax
                    norm = tf.linalg.norm(V, ord='fro',axis=[-2,-1]) 
                    mask = tf.math.divide( tf.math.multiply(norm, ( 0.5 + 0.5 * tf.math.sign( tf.math.square(norm) - self.Pmax ) ) ), tf.math.sqrt(self.Pmax) ) + ( 0.5 + 0.5 * tf.math.sign( self.Pmax - tf.math.square(norm) ) )
                    mask = tf.expand_dims( tf.expand_dims(mask,axis=-1), axis=-1 )
                    V = tf.math.divide( V, mask )
                    self.V = V

            # Final V
            self.pow_alloc = V
            self.pow_alloc1 = tf.math.square(self.pow_alloc)
        
        def U_block(self, V):                        
            # H_ii'V_i
            num = tf.compat.v1.matmul( self.dia, V )
   
            # sigma^2*I + sum_j( (H_ji V_jV_j' H_ji' )
            den = tf.reduce_sum( tf.matmul( self.H, tf.matmul( tf.expand_dims( tf.matmul( V, tf.transpose( V, ( 0,1,3,2 ) ) ), axis=1 ), tf.transpose( self.H, ( 0,1,2,4,3 ) ) ) ), axis=2 ) + self.var
                        
            # U = den^-1 num
            return( tf.compat.v1.matmul( tf.compat.v1.matrix_inverse( den + 1e-4*tf.eye(self.R, dtype=tf.float64) ), num ) )

        # Sum-rate = z
        def W_block(self, U, V):
            # I
            I = tf.eye( self.d, batch_shape=[self.nNodes], dtype=tf.float64 )
            I = tf.tile( tf.expand_dims( I, axis=0 ), tf.constant( [self.batch_size,1,1,1] ) )
            
            # 1 - U_i' H_ii V_i
            den = I - tf.matmul( tf.transpose( U, ( 0,1,3,2 ) ), tf.matmul( self.dia, V ) )
            #pdb.set_trace()
                        
            # W = den^-1
            return( tf.compat.v1.matrix_inverse( den ) )
       
        def V_block(self, U, W):
            # H_ii U_i W_i
            num = tf.matmul( tf.transpose(self.dia, (0,1,3,2)), tf.matmul( U, W ) )
            
            # sum_j( (H_ij' U_j W _j U_j' H_ij )
            den = tf.reduce_sum( tf.matmul( tf.transpose( self.H, ( 0,1,2,4,3 ) ), tf.matmul( tf.expand_dims( tf.matmul( tf.matmul( U, W ), tf.transpose( U, ( 0,1,3,2 ) ) ), axis=2 ), self.H ) ), axis=1 )
                        
            # V = den^-1 num
            return( tf.compat.v1.matmul( tf.compat.v1.matrix_inverse( den + 1e-4*tf.eye(self.T, dtype=tf.float64) ), num ) )        

        def gcn(self, name, out_dim=1):
            # 2 Layers
            L = 2
            
            # Hidden dim = 5
            input_dim = [1, 5]
            output_dim = [5, out_dim]        
            
            ## NSI [Batch_size X Nodes X Features]
            x = tf.ones([self.batch_size, self.nNodes, 1], dtype=tf.float64)

            with tf.variable_scope('gcn_'+name):
                for l in range(L):
                    with tf.compat.v1.variable_scope('gc_l{}'.format(l+1)):
                        # Weights
                        w1 = tf.compat.v1.get_variable( name='w1', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        w0 = tf.compat.v1.get_variable( name='w0', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        
                        ## Biases
                        b1 = tf.compat.v1.get_variable( name='b1', shape=(output_dim[l],), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64 )
                        b0 = tf.compat.v1.get_variable( name='b0', shape=(output_dim[l],), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64 )
                        
                        # XW
                        x1 = tf.matmul(x, w1)
                        x0 = tf.matmul(x, w0)
                        
                        # diag(A)XW0 + AXW1
                        x1 = tf.matmul((self.H_gcn+.001)/1.0, x1)  
                        x0 = tf.matmul((self.dH_gcn+.001)/1.0, x0)
                        
                        ## AXW + B
                        x1 = tf.add(x1, b1)
                        x0 = tf.add(x0, b0)
                        
                        # Combine
                        x = x1 + x0
                        
                        # activation(AXW + B)
                        if l == 0:
                            x = tf.nn.leaky_relu(x)  
                        else:
                            x = tf.nn.sigmoid(x)

                # Output
                output = tf.reshape( x, ( self.batch_size, self.nNodes, out_dim, 1 ) )
            
            return output
                                                                                        
        def build_objective(self):                        
            # H_ii V_i V_i' H_ii'
            num = tf.matmul( tf.matmul( tf.matmul( self.dia, self.pow_alloc ), tf.transpose( self.pow_alloc, (0,1,3,2) ) ), tf.transpose(self.dia,(0,1,3,2) ) )
            
            # sigma^2 + sum_j j ~= i ( (H_ji)^2 * (v_j)^2 ) 
            den = tf.reduce_sum( tf.matmul( self.H, tf.matmul( tf.expand_dims( tf.matmul( self.pow_alloc, tf.transpose( self.pow_alloc, ( 0,1,3,2 ) ) ), axis=1 ), tf.transpose( self.H, ( 0,1,2,4,3 ) ) ) ), axis=2 ) + self.var - num 

            # rate = log(1 + SINR)
            self.rate = tf.math.log( tf.compat.v1.matrix_determinant( tf.eye( self.R, batch_shape=[self.nNodes], dtype=tf.float64 ) + tf.matmul( tf.compat.v1.matrix_inverse( den ), num ) ) ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )
            
            # Sum Rate = sum_i ( rate )
            self.utility = tf.reduce_sum( self.rate, axis=1 )
            util = tf.reduce_mean( self.rate, axis=1 )
            
            # Minimization objective
            self.obj = -tf.reduce_mean( util )
            
            if self.exp == 'uwmmse':
                self.init_optimizer()

        def init_optimizer(self):
            # Gradients and SGD update operation for training the model
            self.trainable_params = tf.compat.v1.trainable_variables()

            # Adam Optimizer
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.obj, self.trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            
            # Update the model
            self.updates = self.opt.apply_gradients(
                zip(clip_gradients, self.trainable_params), global_step=self.global_step)
                
        def save(self, sess, path, var_list=None, global_step=None):
            saver = tf.compat.v1.train.Saver(var_list)
            save_path = saver.save(sess, save_path=path, global_step=global_step)

        def restore(self, sess, path, var_list=None):
            saver = tf.compat.v1.train.Saver(var_list)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(path))

        def train(self, sess, inputs, inp_gcn=None):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            
            output_feed = [self.obj, self.utility, self.pow_alloc1, self.updates]
                            
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]


        def eval(self, sess, inputs, inp_gcn=None):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            
            output_feed = [self.obj,self.utility,self.pow_alloc1]
            
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]
