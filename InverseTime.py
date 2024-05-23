import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from numpy.random import seed
from tensorflow.random import set_seed
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(11)
tf.config.experimental.enable_op_determinism()

parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("-p", "--print_string", nargs='*')
args = parser.parse_args()

print("#################################")
print("dataset: ", args.print_string[0])
print("porcentaje de no etiquetados: ", float(args.print_string[1]))


filename = args.print_string[0]
porc_unlabel = float(args.print_string[1])
#filename = "XJTU"

#porc_unlabel = 0.6




if filename in ["XJTU", "MFPT"]:

  x = np.load("reposit/"+filename+"_data.npy")
  y = np.load("reposit/"+filename+"_label.npy")

elif filename in ["InsectWingbeat"]:
  df =pd.read_csv("reposit2/"+filename+"Sound_TRAIN.tsv", sep="\t", header=None)
  x = df.iloc[:, 1:].values
  y = df.iloc[:, :1].values

elif filename in ["CricketY", "CricketZ"]:

  data_train =pd.read_csv("reposit2/"+filename+"_TRAIN.tsv", sep="\t", header=None)
  data_test =pd.read_csv("reposit2/"+filename+"_TEST.tsv", sep="\t", header=None)
  data = np.concatenate((data_train, data_test))
  x = data[:, 1:]
  y = data[:, 0]

elif filename in ["Yoga", "UWaveGestureLibraryAll", "CricketX", "ECG200"]:

  data_train = np.loadtxt("reposit/"+filename+"_TRAIN.txt")
  data_test = np.loadtxt("reposit/"+filename+"_TEST.txt")
  data = np.concatenate((data_train, data_test))
  x = data[:, 1:]
  y = data[:, 0]

elif filename in ["EpilepticSeizure"]:


  df =pd.read_csv("reposit/"+"EpilepticSeizure.csv")
  x = df.iloc[:, 1:-1].values
  y = df.iloc[:, -1:].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10, stratify=y)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=10, stratify=y_test)



# separo en data etiquetada y no etiquetado

if porc_unlabel > 0:
  x_label, x_unlabel, y_label, y_unlabel = train_test_split(x_train, y_train, test_size=porc_unlabel, random_state=10, stratify=y_train)
else:
  x_label = x_train
  x_unlabel = np.array([])
  y_label = y_train
  y_unlabel = np.array([])


print(x_label.shape, x_unlabel.shape, y_label.shape, y_unlabel.shape)
print(y_label.reshape(-1, 1).shape)
print(y_test.reshape(-1, 1).shape)

# one hot
enc = OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_val, y_test), axis=0).reshape(-1, 1))
y_label = enc.transform(y_label.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

# los no etiquetados llevan 0
y_fake_label = np.zeros(shape=(len(y_unlabel), len(np.unique(y_train))))

# uno etiquetados con no etiquetados

# combino etiquetados con los no
if x_unlabel.shape[0]>0:
  x = np.concatenate((x_label, x_unlabel))
  y = np.concatenate((y_label, y_fake_label))
else:
  x = x_label
  y = y_label

print(x.shape, y.shape)

# uno x con y
train = np.concatenate((x,y), axis=1)
print(train.shape)

# shuffle
np.random.shuffle(train)

# separo
x = train[:, :-len(y_label[0])]
y = train[:, -len(y_label[0]):]
print(x.shape, y.shape)

# reshape
x = np.reshape(x, [x.shape[0], x.shape[1], 1])
x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], 1])
x_val = np.reshape(x_val, [x_val.shape[0], x_val.shape[1], 1])
y_true = np.argmax(y_test, axis=1)
print(x.shape, y.shape)
print(x_test.shape, y_test.shape)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

#x_flip = shuffle_along_axis(x, 1)


x_flip = np.flip(x,1)


y_fake_ceros = np.zeros(shape=(len(x),1))
y_fake_ones = np.ones(shape=(len(x),1))

y_fake = np.zeros(shape=y.shape)

new_x = np.concatenate((x, x_flip))
y_kmeans_onehot =  np.concatenate((y_fake_ones, y_fake_ceros))
y_fake_ones_val = np.ones(shape=(len(x_val),1))



accuracy = []
recall = []
precision = []

alpha = 0.9

for random_seed in range(10):
  print("model:", random_seed)
  tf.keras.utils.set_random_seed(random_seed)
  tf.config.experimental.enable_op_determinism()


  cnn_input = tf.keras.Input(shape=(x.shape[1], 1))


  conv1 = keras.layers.Conv1D(filters=16, kernel_size=8, padding='same')(cnn_input)
  conv1 = keras.layers.BatchNormalization()(conv1)
  conv1 = keras.layers.Activation('relu')(conv1)

  conv2 = keras.layers.Conv1D(filters=16, kernel_size=5, padding='same')(conv1)
  conv2 = keras.layers.BatchNormalization()(conv2)
  conv2 = keras.layers.Activation('relu')(conv2)

  conv3 = keras.layers.Conv1D(filters=32, kernel_size=3, padding='same')(conv2)
  conv3 = keras.layers.BatchNormalization()(conv3)
  conv3 = keras.layers.Activation('relu')(conv3)

  conv4 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv3)
  conv4 = keras.layers.BatchNormalization()(conv4)
  conv4 = keras.layers.Activation('relu')(conv4)

  # FINAL

  gap_layer = keras.layers.GlobalAveragePooling1D()(conv4)

  model_cnn = keras.models.Model(cnn_input, gap_layer)


  classifier_input = tf.keras.Input(shape=(2*32,))
  classifier_output = tf.keras.layers.Dense(len(np.unique(y_true)), activation="softmax")(classifier_input)
  classifier = tf.keras.Model(classifier_input, classifier_output, name="classifier")

  pretext_input = tf.keras.Input(shape=(2*32,))
  pretext_output = tf.keras.layers.Dense(1, activation="softmax")(pretext_input)
  pretext_classifier = tf.keras.Model(pretext_input, pretext_output, name="pretext_classifier")

  cnn_classif_input = tf.keras.Input(shape=(x.shape[1], 1), name="cnn")
  cnn_series = model_cnn(cnn_classif_input)
  classif_series = classifier(cnn_series)
  pretext_series = pretext_classifier(cnn_series)

  final_classif2 = tf.keras.Model(cnn_classif_input, outputs=[classif_series,pretext_series], name="CnnClf")

  final_classif2.summary()

  def my_loss_sup(y_truee, y_pred):

    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    l1 = cce(y_truee, y_pred)
    mask = tf.reduce_sum(y_truee,axis=1)#
    mloss = tf.multiply(mask,l1)
    mloss = tf.reduce_mean(mloss)#/tf.math.maximum(tf.reduce_sum(mask),1))

    return mloss

  def my_loss_unsup(y_trux, y_pred):
    cce2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    local_y = y_trux[:,:len(np.unique(y_true))]
    local_kmeans_y = y_trux[:,len(np.unique(y_true)):]

    l2 = cce2(local_kmeans_y, y_pred)
    mask2 = 1.0-tf.reduce_sum(local_y,axis=1)
    mloss2 = tf.multiply(mask2,l2)
    mloss2 = alpha*tf.reduce_mean(mloss2)

    return  mloss2

  final_classif2.compile(
  loss = [my_loss_sup, my_loss_unsup],
  optimizer= tf.keras.optimizers.Adam())

  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
  model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="best2.h5", monitor='val_loss',
                                                    save_best_only=True)
  callbacks = [reduce_lr, model_checkpoint]
  y_ = np.concatenate((y,y_fake))
  y_segunda_loss = np.concatenate((y_, y_kmeans_onehot),axis=1)

  y_segunda_loss_val = np.concatenate((y_val, y_fake_ones_val),axis=1)

  final_classif2.fit(new_x, [y_, y_segunda_loss], verbose=2, epochs=1000, batch_size=None,
                    callbacks=callbacks, validation_data=(x_val, [y_val, y_segunda_loss_val]))

  best2 = keras.models.load_model("best2.h5", compile=False)
  y_pred = best2.predict(x_test)[0]

  y_pred = np.argmax(y_pred, axis=1)

  acc = accuracy_score(y_true, y_pred)
  sco = precision_score(y_true, y_pred, average='macro')
  rec = recall_score(y_true, y_pred, average='macro')

  accuracy.append(acc)
  recall.append(rec)
  precision.append(sco)
  print(sco, acc, rec)

res1 = pd.DataFrame({"precision": precision, "accuracy": accuracy, "recall": recall})
print(res1)
print(res1.describe())

res1.to_csv("2_new_results_invers/random_time_"+filename+"_"+str(porc_unlabel)+"_alpha2="+str(alpha)+".csv")
