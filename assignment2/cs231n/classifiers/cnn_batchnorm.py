import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class Enhanced_CNN(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 16], filter_size=[9, 5],
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               use_batchnorm=False, dropout=0, relu_leakiness=0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.use_batchnorm = use_batchnorm
    self.dtype = dtype
    self.use_dropout = False if dropout == 0 else True
    self.dropout = dropout
    self.relu_leakiness = relu_leakiness
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['W1'] = np.random.randn(num_filters[0],
                                        input_dim[0],
                                        filter_size[0],
                                        filter_size[0]) * weight_scale
    self.params['b1'] = np.zeros(num_filters[0])
    
    if use_batchnorm == True:
        self.params['gamma1'] = np.ones(num_filters[0])
        self.params['beta1'] = np.zeros(num_filters[0])
        self.params['gamma2'] = np.ones(num_filters[1])
        self.params['beta2'] = np.zeros(num_filters[1])
        self.params['gamma3'] = np.ones(hidden_dim)
        self.params['beta3'] = np.zeros(hidden_dim)

    self.params['W2'] = np.random.randn(num_filters[1],
                                        num_filters[0],
                                        filter_size[1],
                                        filter_size[1]) * weight_scale
    self.params['b2'] = np.zeros(num_filters[1])

    self.params['W3'] = np.random.randn(num_filters[1] * input_dim[1] / 2 * input_dim[2] / 2,
                                        hidden_dim) * weight_scale
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['W4'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b4'] = np.zeros(num_classes)

    if use_batchnorm == True:
        self.affine_bn_param = {'mode': 'train'}
        self.spatial_bn_param = [{'mode': 'train'} for i in range(2)]
    if self.use_dropout:
        self.dropout_param = {'p': self.dropout, 'mode': 'train'}

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']

    if self.use_batchnorm == True:
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = [W1.shape[2], W2.shape[2]]
    conv1_param = {'stride': 1, 'pad': (filter_size[0] - 1) / 2}
    conv2_param = {'stride': 1, 'pad': (filter_size[1] - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    mode = 'test' if y is None else 'train'

    # pass bn_param to the forward pass for the spatial batchnorm layer
    if self.use_batchnorm == True:
        spatial_bn_param = self.spatial_bn_param
        affine_bn_param = self.affine_bn_param
        for param in spatial_bn_param:
            param['mode'] = mode
        affine_bn_param['mode'] = mode

    if self.use_dropout:
        dropout_param = self.dropout_param
        dropout_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ###########################################################################
    
    if self.use_batchnorm == True:
        conv1_out, conv1_cache = conv_bn_relu_forward(X, W1, b1,
                                                      gamma1, beta1,
                                                      self.relu_leakiness,
                                                      conv1_param, spatial_bn_param[0])
        if self.use_dropout == True:
            conv1_out, conv1_dropout_cache = dropout_forward(conv1_out, dropout_param)
        conv2_out, conv2_cache = conv_bn_relu_forward(conv1_out, W2, b2,
                                                      gamma2, beta2,
                                                      self.relu_leakiness,
                                                      conv2_param, spatial_bn_param[1])
        if self.use_dropout:
            conv2_out, conv2_dropout_cache = dropout_forward(conv2_out, dropout_param)
    else:
        conv1_out, conv1_cache = conv_relu_forward(X, W1, b1, 
                                                 conv1_param)
        if self.use_dropout:
            conv1_out, conv1_dropout_cache = dropout_forward(conv1_out, dropout_param)
        conv2_out, conv2_cache = conv_relu_forward(conv1_out, W2, b2,
                                                   conv2_param)
        if self.use_dropout:
            conv2_out, conv2_dropout_cache = dropout_forward(conv2_out, dropout_param)

    pool_out, pool_cache = max_pool_forward_fast(conv2_out, pool_param)

    ar_out, ar_cache = affine_relu_forward(pool_out, W3, b3, self.relu_leakiness)

    if self.use_batchnorm == True:
        ar_out, bn_cache = batchnorm_forward(ar_out, gamma3, beta3, affine_bn_param)
    if self.use_dropout:
        ar_out, affine_dropout_cache = dropout_forward(ar_out, dropout_param)

    affine_out, affine_cache = affine_forward(ar_out, W4, b4)
    scores = affine_out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + \
            np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2) + \
            np.sum(self.params['W4'] ** 2))

    dx, dw, db = affine_backward(dout, affine_cache)
    grads['W4'] = dw + self.reg * self.params['W4']
    grads['b4'] = db

    if self.use_dropout:
        dx = dropout_backward(dx, affine_dropout_cache)
    if self.use_batchnorm == True:
        dx, dgamma, dbeta = batchnorm_backward_alt(dx, bn_cache)
        grads['gamma3'] = dgamma
        grads['beta3'] = dbeta

    dx, dw, db = affine_relu_backward(dx, ar_cache)
    grads['W3'] = dw + self.reg * self.params['W3']
    grads['b3'] = db

    dx = max_pool_backward_fast(dx, pool_cache)

    if self.dropout:
        dx = dropout_backward(dx, conv2_dropout_cache)
    if self.use_batchnorm == True:
        dx, dw, db, dgamma, dbeta = conv_bn_relu_backward(dx, conv2_cache)
        grads['gamma2'] = dgamma
        grads['beta2'] = dbeta
    else:
        dx, dw, db = conv_relu_backward(dx, conv2_cache)
    grads['W2'] = dw + self.reg * self.params['W2']
    grads['b2'] = db

    if self.dropout:
        dx = dropout_backward(dx, conv1_dropout_cache)
    if self.use_batchnorm:
        dx, dw, db, dgamma, dbeta = conv_bn_relu_backward(dx, conv1_cache)
        grads['gamma1'] = dgamma
        grads['beta1'] = dbeta
    else:
        dx, dw, db = conv_relu_backward(dx, conv1_cache)

    grads['W1'] = dw + self.reg * self.params['W1']
    grads['b1'] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
