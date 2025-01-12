import tensorflow as tf
import numpy as np

def ReLU(x):
    return tf.maximum(0.0, x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Tanh(x):
    return tf.tanh(x)

def Iden(x):
    return x

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None, use_bias=False):
        self.input = input
        self.activation = activation

        if W is None:
            if activation.__name__ == "ReLU":
                W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=np.float32)
            else:
                W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ), dtype=np.float32)
            W = tf.Variable(W_values, name='W')
        if b is None:
            b_values = np.zeros((n_out,), dtype=np.float32)
            b = tf.Variable(b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = tf.matmul(input, self.W) + self.b
        else:
            lin_output = tf.matmul(input, self.W)

        self.output = lin_output if activation is None else activation(lin_output)

        # Parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

def _dropout_from_layer(rng, layer, p):
    mask = tf.random.uniform(layer.shape, minval=0, maxval=1) < (1 - p)
    return tf.multiply(layer, tf.cast(mask, tf.float32)) / (1 - p)

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
            activation=activation, use_bias=use_bias
        )
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

class MLPDropout(object):
    def __init__(self, rng, input, layer_sizes, dropout_rates, activations, use_bias=True):
        self.weight_matrix_sizes = list(zip(layer_sizes, layer_sizes[1:]))
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        next_layer_input = input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0

        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(
                rng=rng, input=next_dropout_layer_input,
                activation=activations[layer_counter],
                n_in=n_in, n_out=n_out, use_bias=use_bias,
                dropout_rate=dropout_rates[layer_counter]
            )
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            next_layer = HiddenLayer(
                rng=rng, input=next_layer_input,
                activation=activations[layer_counter],
                W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                b=next_dropout_layer.b,
                n_in=n_in, n_out=n_out, use_bias=use_bias
            )
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            layer_counter += 1

        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
            input=next_dropout_layer_input,
            n_in=n_in, n_out=n_out
        )
        self.dropout_layers.append(dropout_output_layer)

        output_layer = LogisticRegression(
            input=next_layer_input,
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out
        )
        self.layers.append(output_layer)

        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        self.params = [param for layer in self.dropout_layers for param in layer.params]

    def predict(self, new_data):
        next_layer_input = new_data
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_layer_input = self.activations[i](tf.matmul(next_layer_input, layer.W) + layer.b)
            else:
                p_y_given_x = tf.nn.softmax(tf.matmul(next_layer_input, layer.W) + layer.b)
        y_pred = tf.argmax(p_y_given_x, axis=1)
        return y_pred

    def predict_p(self, new_data):
        next_layer_input = new_data
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_layer_input = self.activations[i](tf.matmul(next_layer_input, layer.W) + layer.b)
            else:
                p_y_given_x = tf.nn.softmax(tf.matmul(next_layer_input, layer.W) + layer.b)
        return p_y_given_x

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng, input=input,
            n_in=n_in, n_out=n_hidden,
            activation=Tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden, n_out=n_out
        )
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            self.W = tf.Variable(np.zeros((n_in, n_out), dtype=np.float32), name='W')
        else:
            self.W = W
        if b is None:
            self.b = tf.Variable(np.zeros((n_out,), dtype=np.float32), name='b')
        else:
            self.b = b

        self.p_y_given_x = tf.nn.softmax(tf.matmul(input, self.W) + self.b)
        self.y_pred = tf.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -tf.reduce_mean(tf.math.log(self.p_y_given_x)[tf.arange(y.shape[0]), y])

    def errors(self, y):
        if y.dtype != tf.int32:
            y = tf.cast(y, tf.int32)
        return tf.reduce_mean(tf.cast(tf.not_equal(self.y_pred, y), tf.float32))

class LeNetConvPoolLayer(object):
    def __init__(self, rng, filter_shape, poolsize=(2, 2), non_linear="tanh"):
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = tf.Variable(tf.random.uniform(filter_shape, minval=-0.01, maxval=0.01), name="W_conv")
        else:
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = tf.Variable(tf.random.uniform(filter_shape, minval=-W_bound, maxval=W_bound), name="W_conv")

        self.b = tf.Variable(tf.zeros((filter_shape[0],)), name="b_conv")
        self.params = [self.W, self.b]

    def set_input(self, input):
        conv_out = tf.nn.conv2d(input, self.W, strides=[1, 1, 1, 1], padding="VALID")
        if self.non_linear == "tanh":
            conv_out_tanh = tf.tanh(conv_out + self.b)
            output = tf.nn.max_pool(conv_out_tanh, ksize=[1, *self.poolsize, 1], strides=[1, *self.poolsize, 1], padding="VALID")
        elif self.non_linear == "relu":
            conv_out_tanh = tf.nn.relu(conv_out + self.b)
            output = tf.nn.max_pool(conv_out_tanh, ksize=[1, *self.poolsize, 1], strides=[1, *self.poolsize, 1], padding="VALID")
        else:
            pooled_out = tf.nn.max_pool(conv_out, ksize=[1, *self.poolsize, 1], strides=[1, *self.poolsize, 1], padding="VALID")
            output = pooled_out + self.b
        return output