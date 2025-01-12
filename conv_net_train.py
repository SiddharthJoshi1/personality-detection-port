"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Migrated from Theano to TensorFlow 2.x/Keras
"""
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import warnings
import sys
import time
import csv
warnings.filterwarnings("ignore")

# Different non-linearities
def ReLU(x):
    return tf.maximum(0.0, x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Tanh(x):
    return tf.tanh(x)

def Iden(x):
    return x

def train_conv_net(datasets,
                   U,
                   ofile,
                   cv=0,
                   attr=0,
                   img_w=312,  # Updated to match the word vector length in the dataset
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay=0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (312 for the dataset)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0][0])  # Sentence length (padded)
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))

    parameters = [("image shape", img_h, img_w), ("filter shape", filter_shapes), ("hidden_units", hidden_units),
                  ("dropout", dropout_rate), ("batch_size", batch_size), ("non_static", non_static),
                  ("learn_decay", lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static),
                  ("sqr_norm_lim", sqr_norm_lim), ("shuffle_batch", shuffle_batch)]
    print(parameters)

    # Define model architecture
    input_shape = (img_h, img_w, 1)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    conv_layers = []
    for i in range(len(filter_hs)):
        conv_layer = layers.Conv2D(feature_maps, filter_shapes[i], activation=conv_non_linear)(x)
        pool_layer = layers.MaxPooling2D(pool_size=pool_sizes[i])(conv_layer)
        conv_layers.append(pool_layer)

    x = layers.concatenate(conv_layers, axis=-1)
    x = layers.Flatten()(x)

    # Add Mairesse features
    mair_input = tf.keras.Input(shape=(datasets[4].shape[1],))
    x = layers.concatenate([x, mair_input])
    hidden_units[0] = feature_maps * len(filter_hs) + datasets[4].shape[1]

    for i, (units, activation) in enumerate(zip(hidden_units, activations)):
        x = layers.Dense(units, activation=activation)(x)
        if i < len(dropout_rate):
            x = layers.Dropout(dropout_rate[i])(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=[inputs, mair_input], outputs=outputs)

    # Compile the model
    optimizer = optimizers.Adadelta(learning_rate=1.0, rho=lr_decay, epsilon=1e-6)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Prepare data
    train_set_x, train_set_y, train_set_m = datasets[0], datasets[1], datasets[4]
    val_set_x, val_set_y, val_set_m = datasets[0][-int(0.1 * len(datasets[0])):], datasets[1][-int(0.1 * len(datasets[1])):], datasets[4][-int(0.1 * len(datasets[4])):]
    test_set_x, test_set_y, test_set_m = datasets[2], datasets[3], datasets[5]

    # Debugging: Print shapes before reshaping
    print(f"Original train_set_x shape: {train_set_x.shape}")
    print(f"Expected train_set_x shape: (num_samples, {img_h}, {img_w})")

    # Fix shape mismatch
    if train_set_x.shape[1] != img_h or train_set_x.shape[2] != img_w:
        # Transpose the data to match the expected shape
        train_set_x = np.transpose(train_set_x, (0, 2, 1))  # Swap the second and third dimensions
        print(f"Transposed train_set_x shape: {train_set_x.shape}")

    # Reshape data for Conv2D
    train_set_x = np.expand_dims(train_set_x, axis=-1)  # Add channel dimension
    print(f"Reshaped train_set_x shape: {train_set_x.shape}")

    val_set_x = np.transpose(val_set_x, (0, 2, 1))
    val_set_x = np.expand_dims(val_set_x, axis=-1)

    test_set_x = np.transpose(test_set_x, (0, 2, 1))
    test_set_x = np.expand_dims(test_set_x, axis=-1)

    # Train the model
    history = model.fit([train_set_x, train_set_m], train_set_y,
                        validation_data=([val_set_x, val_set_m], val_set_y),
                        epochs=n_epochs, batch_size=batch_size, shuffle=shuffle_batch)

    # Evaluate the model
    test_loss, test_acc = model.evaluate([test_set_x, test_set_m], test_set_y, batch_size=batch_size)
    print(f"Test accuracy: {test_acc}")

    return test_acc, history.history['val_accuracy'][-1]

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables """
    data_x, data_y, data_m = data_xy
    shared_x = tf.convert_to_tensor(np.asarray(data_x, dtype=np.float32))
    shared_y = tf.convert_to_tensor(np.asarray(data_y, dtype=np.int32))
    shared_m = tf.convert_to_tensor(np.asarray(data_m, dtype=np.float32))
    return shared_x, shared_y, shared_m

def get_idx_from_sent(status, word_idx_map, charged_words, max_l=51, max_s=200, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    length = len(status)

    pass_one = True
    while len(x) == 0:
        for i in range(length):
            words = status[i].split()
            if pass_one:
                words_set = set(words)
                if len(charged_words.intersection(words_set)) == 0:
                    continue
            else:
                if np.random.randint(0, 2) == 0:
                    continue
            y = []
            for _ in range(pad):
                y.append(0)
            for word in words:
                if word in word_idx_map:
                    y.append(word_idx_map[word])

            while len(y) < max_l + 2 * pad:
                y.append(0)
            x.append(y)
        pass_one = False

    if len(x) < max_s:
        x.extend([[0] * (max_l + 2 * pad)] * (max_s - len(x)))

    return x

def make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, cv, per_attr=0, max_l=51, max_s=200, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainX, testX, trainY, testY, mTrain, mTest = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, charged_words, max_l, max_s, k, filter_h)

        if rev["split"] == cv:
            testX.append(sent)
            testY.append(rev['y' + str(per_attr)])
            mTest.append(mairesse[rev["user"]])
        else:
            trainX.append(sent)
            trainY.append(rev['y' + str(per_attr)])
            mTrain.append(mairesse[rev["user"]])
    trainX = np.array(trainX, dtype="int32")
    testX = np.array(testX, dtype="int32")
    trainY = np.array(trainY, dtype="int32")
    testY = np.array(testY, dtype="int32")
    mTrain = np.array(mTrain, dtype=np.float32)
    mTest = np.array(mTest, dtype=np.float32)
    return [trainX, trainY, testX, testY, mTrain, mTest]

if __name__ == "__main__":
    print("loading data...")
    x = pickle.load(open("essays_mairesse.p", "rb"))
    revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
    print("data loaded!")
    mode = sys.argv[1]
    word_vectors = sys.argv[2]
    attr = int(sys.argv[3])
    if mode == "-nonstatic":
        print("model architecture: CNN-non-static")
        non_static = True
    elif mode == "-static":
        print("model architecture: CNN-static")
        non_static = False

    if word_vectors == "-rand":
        print("using: random vectors")
        U = W2
    elif word_vectors == "-word2vec":
        print("using: word2vec vectors")
        U = W

    r = range(0, 10)

    ofile = open('perf_output_' + str(attr) + '.txt', 'w')

    charged_words = []

    emof = open("Emotion_Lexicon.csv", "r")
    csvf = csv.reader(emof, delimiter=',', quotechar='"')
    first_line = True

    for line in csvf:
        if first_line:
            first_line = False
            continue
        if line[11] == "1":
            charged_words.append(line[0])

    emof.close()

    charged_words = set(charged_words)

    results = []
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, i, attr, max_l=149, max_s=312, k=300, filter_h=3)

        perf, fscore = train_conv_net(datasets,
                                      U,
                                      ofile,
                                      cv=i,
                                      attr=attr,
                                      lr_decay=0.95,
                                      filter_hs=[1, 2, 3],
                                      conv_non_linear="relu",
                                      hidden_units=[200, 200, 2],
                                      shuffle_batch=True,
                                      n_epochs=50,
                                      sqr_norm_lim=9,
                                      non_static=non_static,
                                      batch_size=50,
                                      dropout_rate=[0.5, 0.5, 0.5],
                                      activations=[Sigmoid])
        output = "cv: " + str(i) + ", perf: " + str(perf) + ", macro_fscore: " + str(fscore)
        print(output)
        ofile.write(output + "\n")
        ofile.flush()
        results.append([perf, fscore])
    results = np.asarray(results)
    perf_out = 'Perf : ' + str(np.mean(results[:, 0]))
    fscore_out = 'Macro_Fscore : ' + str(np.mean(results[:, 1]))
    print(perf_out)
    print(fscore_out)
    ofile.write(perf_out + "\n" + fscore_out)
    ofile.close()