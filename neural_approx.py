import numpy as np

from tqdm import tqdm

import tensorflow.compat.v1 as tf
real_type = tf.float32

from preprocessing import normalize_data, diff_PCA


def vanilla_net(
    input_dim, #dim of inputs
    hidden_units, #units in hidden layers (assumed const, but doesn't need to be this way)
    hidden_layers, #number of hidden layers
    seed): #seed for initialization, or None for random

    
    #input layer
    xs = tf.placeholder(shape = [None, input_dim], dtype = real_type)

    #weights and biases of hidden layers
    ws = [None]
    bs = [None]
    #Note: layer 0 (input layer) has no params

    #layer 0 = input layer, identity
    zs = [xs]

    ##### first hidden layer (index 1) to initialize #####
    #weight matrix
    ws.append(tf.get_variable("w1",
                              [input_dim, hidden_units],
                              initializer = tf.variance_scaling_initializer(),
                              dtype = real_type))
    #bias vector
    bs.append(tf.get_variable("b1",
                              [hidden_units],
                              initializer = tf.zeros_initializer(),
                              dtype = real_type))
    
    #graph (just lang for how to have params)
    zs.append(zs[0] @ ws[1] + bs[1])

    ##### for second layer onwards #####
    for l in range(1,hidden_layers):
        ws.append(tf.get_variable(f"w{l+1}",
                                  [hidden_units, hidden_units],
                                  initializer = tf.variance_scaling_initializer(),
                                  dtype = real_type))
        bs.append(tf.get_variable(f"b{l+1}",
                                  [hidden_units],
                                  initializer = tf.zeros_initializer(),
                                  dtype = real_type))
        zs.append(tf.nn.softplus(zs[l]) @ ws[l+1] + bs[l+1])

    ##### output layer #####
    #weight matrix
    ws.append(tf.get_variable(f"w{hidden_layers+1}",
                              [hidden_units, 1],
                              initializer = tf.variance_scaling_initializer(),
                              dtype = real_type))
    #bias vector
    bs.append(tf.get_variable(f"b{hidden_layers+1}",
                              [1],
                              initializer = tf.zeros_initializer(),
                              dtype = real_type))

    #graph (just lang for how to have params)
    zs.append(tf.nn.softplus(zs[hidden_layers]) @ ws[hidden_layers+1] + bs[hidden_layers+1])

    #result is the last output layer
    ys = zs[hidden_layers+1]#tf.nn.relu(zs[hidden_layers+1]) - y_mean_global / y_std_global#tf.nn.relu(zs[hidden_layers+1]) - 0.03978625283374658 / 0.06264915839363691#tf.nn.relu(zs[hidden_layers+1]) - 0.03978625283374658 / 0.06264915839363691 #relu seems to work quite well
    return xs, (ws, bs), zs, ys


def backprop(weights_and_biases, #2nd output vanilla_net()
             zs,
             ys): #3nd output vanilla_net()

    ws, bs = weights_and_biases
    L = len(zs) - 1

    # backprop
    zbar = tf.ones_like(zs[L])#tf.nn.relu(tf.math.sign(zs[L]))#tf.nn.sigmoid(ys) + tf.nn.silu(ys) * tf.nn.sigmoid(-ys)#tf.nn.sigmoid(ys)#tf.ones_like(zs[L])#
    for l in range(L - 1, 0, -1): #iterate backwards
        zbar = (zbar @ tf.transpose(ws[l+1])) * tf.nn.sigmoid(zs[l])

    #for l = 0
    zbar = zbar @ tf.transpose(ws[1])

    xbar = zbar

    return xbar


def twin_net(input_dim,
             hidden_units,
             hidden_layers,
             seed):
    #first half is vanilla net
    xs, (ws, bs), zs, ys = vanilla_net(input_dim,
                                       hidden_units,
                                       hidden_layers,
                                       seed)
    #second half is backprop
    xbar = backprop((ws, bs), zs, ys)
    
    return xs, ys, xbar


###################################################################
######################### VANILLLA SETUP ##########################

def vanilla_training_graph(input_dim,
                           hidden_units,
                           hidden_layers,
                           seed):
    
    #net
    inputs, weights_and_biases, layers, predictions = \
        vanilla_net(input_dim, hidden_units, hidden_layers, seed)

    #standard backprop
    derivs_predictions = backprop(weights_and_biases, layers, predictions)

    # placeholder for labels
    labels = tf.placeholder(shape = [None, 1], dtype = real_type)

    # loss function
    loss = tf.losses.mean_squared_error(labels, predictions)

    # optimizer
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    return inputs, labels, predictions, derivs_predictions, learning_rate, loss, optimizer.minimize(loss)


def vanilla_train_one_epoch(inputs,
                        labels,
                        lr_placeholder,
                        minimizer,
                        x_train,
                        y_train,
                        learning_rate,
                        batch_size,
                        session):
    m, n = x_train.shape

    # minimization loop over mini-batches
    first = 0
    last = min(batch_size, m)
    
    while first < m:
        session.run(minimizer,
                    feed_dict = {
                        inputs: x_train[first:last],
                        labels: y_train[first:last],
                        lr_placeholder: learning_rate
                    })
        first = last
        last = min(first + batch_size, m)

###################################################################

###################################################################
####################### DIFFERENTIAL SETUP ########################

def differential_training_graph(input_dim,
                                hidden_units,
                                hidden_layers,
                                seed,
                                alpha,
                                beta,
                                lambda_j):
    """
    loss = alpha*MSE(values) + beta*MSE(greeks, lambda_j)
    """
    inputs, predictions, derivs_predictions = twin_net(input_dim,
                                                       hidden_units,
                                                       hidden_layers,
                                                       seed)
    
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    derivs_labels = tf.placeholder(shape=[None, derivs_predictions.shape[1]], dtype=real_type)
    
    loss = alpha * tf.losses.mean_squared_error(labels, predictions) + \
            beta * tf.losses.mean_squared_error(derivs_labels*lambda_j, derivs_predictions*lambda_j)
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    return inputs, labels, derivs_labels, predictions, derivs_predictions, learning_rate, loss, optimizer.minimize(loss)


def differential_train_one_epoch(inputs,
                         labels,
                         derivs_labels,
                         lr_placeholder,
                         minimizer,
                         x_train,
                         y_train,
                         dydx_train,
                         learning_rate,
                         batch_size,
                         session):
    m, n = x_train.shape
    
    first = 0
    last = min(batch_size, m)

    while first < m:
        session.run(minimizer,
                    feed_dict = {
                        inputs: x_train[first:last],
                        labels: y_train[first:last],
                        derivs_labels: dydx_train[first:last],
                        lr_placeholder: learning_rate
                    })
        first = last
        last = min(first+batch_size, m)
        
###################################################################

def train(description,
          approximator, #neural approximator
          reinit=True,
          epochs = 10,
          learning_rate_schedule = [(0.0, 1.0e-8),
                                    (0.2, 0.1),
                                    (0.6, 0.01),
                                    (0.9, 1.0e-6),
                                    (1.0, 1.0e-8)],
          batches_per_epoch=16,
          min_batch_size=256,
          callback=None,
          callback_epochs=[]
          ):
  
  
    batch_size = max(min_batch_size, approximator.m // batches_per_epoch)

    lr_schedule_epochs, lr_schedule_rates = zip(*learning_rate_schedule)

    # reset
    if reinit:
        approximator.session.run(approximator.initializer)
    
    # callback on epoch 0 (if requested)
    if callback and 0 in callback_epochs:
        callback(approximator, 0)
  
    for epoch in tqdm(range(epochs), desc=description):
        learning_rate = np.interp(epoch/epochs, lr_schedule_epochs, lr_schedule_rates)

        if not approximator.differential: #träna vanlig, till skillnad från differential
            vanilla_train_one_epoch(
                approximator.inputs,
                approximator.labels,
                approximator.learning_rate,
                approximator.minimizer,
                approximator.x,
                approximator.y,
                learning_rate, #does not belong to approximator
                batch_size, #does not belong to approximator
                approximator.session)
        else:
            differential_train_one_epoch(
                approximator.inputs,
                approximator.labels,
                approximator.derivs_labels,
                approximator.learning_rate,
                approximator.minimizer,
                approximator.x,
                approximator.y,
                approximator.dy_dx,
                learning_rate, #does not belong to approximator
                batch_size, #does not belong to approximator
                approximator.session)
        
        # callback in loop (if requested)
        if callback and epoch in callback_epochs:
            callback(approximator, epoch)
            print(f"===== epoch: {epoch} =====")

    # final callback (if requested)
    if callback and epochs in callback_epochs:
        callback(approximator, epochs)


class NeuralApproximator():
    def __init__(self, x_raw, y_raw, dydx_raw = None, eps_PCA = None, eps_diff_PCA = None, eps_diff_fixed_dim = None):
        """
        NOTE: If using eps_diff_fixed_dim, need to set eps_diff_PCA != 0, even though it is not used
        """
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
        self.eps_PCA = eps_PCA # near numerical zero
        self.eps_diff_PCA = eps_diff_PCA
        self.eps_diff_fixed_dim  = eps_diff_fixed_dim

        self.graph = None #tf logic
        self.session = None #tf logic
        

    def __del__(self): #finalizer, run when garbage collection
        if self.session is not None:
            self.session.close()

    def build_graph(self,
                    differential, #boolean
                    lam, #balance 
                    hidden_units,
                    hidden_layers,
                    weight_seed):
        
        #tensor flow logic:
        if self.session is not None:
            self.session.close()
        
        self.graph = tf.Graph()

        with self.graph.as_default():

            #build the graph: vanilla or differential
            self.differential = differential

            if not differential: #vanilla

                self.inputs, \
                self.labels, \
                self.predictions, \
                self.derivs_predictions, \
                self.learning_rate, \
                self.loss, \
                self.minimizer \
                = vanilla_training_graph(self.n,
                                         hidden_units,
                                         hidden_layers,
                                         weight_seed)
            else:
                if self.dy_dx is None:
                    raise Exception("No differential labels for differential training graph")
                self.alpha = 1.0/(1.0 + lam * self.n)
                self.beta = 1.0 - self.alpha
                
                self.inputs, \
                self.labels, \
                self.derivs_labels, \
                self.predictions, \
                self.derivs_predictions, \
                self.learning_rate, \
                self.loss, \
                self.minimizer \
                = differential_training_graph(self.n,
                                        hidden_units,
                                        hidden_layers,
                                        weight_seed,
                                        self.alpha,
                                        self.beta,
                                        self.lambda_j)

            #global initializer
            self.initializer = tf.global_variables_initializer()

        #done
        self.graph.finalize()
        self.session = tf.Session(graph=self.graph)

    def prepare(self,
                m,
                differential,
                lam = 1, #balance cost between values and derivs
                hidden_units = 20, #std architecture
                hidden_layers = 4, #std architecture
                weight_seed = None):
        
        # prepare dataset, x and y and dydx is normalised
        if self.eps_PCA is not None and self.eps_diff_PCA is None:
            raise Exception("Ordinary PCA not implemented")
        elif self.eps_diff_PCA is not None:
            self.x, \
            self.y, \
            self.dy_dx, \
            self.x_mean, \
            self.y_mean, \
            self.y_std, \
            self.P_2_tilde, \
            self.D_2_tilde_sqrt_inv, \
            self.P_3_tilde, \
            self.lambda_j \
            = diff_PCA(self.x_raw,
                       self.y_raw,
                       self.dydx_raw,
                       self.eps_PCA, 
                       self.eps_diff_PCA,
                       self.eps_diff_fixed_dim,
                       m)
            self.m, self.n = self.x.shape
            
            global y_mean_global, y_std_global
            y_mean_global = self.y_mean
            y_std_global = self.y_std

            self.build_graph(differential,
                            lam,
                            hidden_units,
                            hidden_layers,
                            weight_seed)
        else:
            self.x_mean, \
            self.x_std, \
            self.x, \
            self.y_mean, \
            self.y_std, \
            self.y, \
            self.dy_dx, \
            self.lambda_j \
            = normalize_data(self.x_raw,
                            self.y_raw,
                            self.dydx_raw,
                            m)

        self.m, self.n = self.x.shape

        self.build_graph(differential,
                        lam,
                        hidden_units,
                        hidden_layers,
                        weight_seed)

    def train(self,
              description = '=== TRAINING ===',
              reinit = True,
              epochs = 100,
              learning_rate_schedule = [(0.0, 1.0e-8),
                                        (0.2, 0.1),
                                        (0.6, 0.01),
                                        (0.9, 1.0e-6),
                                        (1.0, 1.0e-8)],
              batches_per_epoch = 16,
              min_batch_size = 256,
              callback = None, #callbacks are not used, but good for debugging
              callback_epochs = []):
    
        train(description,
            self,
            reinit,
            epochs,
            learning_rate_schedule,
            batches_per_epoch,
            min_batch_size,
            callback,
            callback_epochs)
  
    def predict_values(self, x):
        
        if self.eps_diff_PCA is not None:
            # diffPCA
            # Transform inputs
            x_1 = x - self.x_mean
            x_3 = x_1 @ self.P_2_tilde @ self.D_2_tilde_sqrt_inv @ self.P_3_tilde
            y_scaled = self.session.run(self.predictions,
                                        feed_dict = {self.inputs: x_3})
            y = self.y_mean + self.y_std * y_scaled
            
        elif self.eps_PCA is not None:
            print("normal PCA not implemented")
        else:
            # No PCA nor diffPCA
            x_scaled = (x - self.x_mean) / self.x_std
            y_scaled = self.session.run(self.predictions,
                                        feed_dict = {self.inputs: x_scaled})
        
            #unscale
            y = y_scaled * self.y_std + self.y_mean
        
        return y