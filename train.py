import tensorflow as tf
from tensorflow import keras
import numpy as np
#import sys
import os

tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(6)

'''if len(sys.argv) <= 1:
    print("Please specify initial epoch; -1 for first train")
    sys.exit(-1)'''

seeds = 42
n_steps = 100
batch_size = 1
embed_dim = 10
rnn_cells = 128
filepath = "data/merged.txt"
checkpoint_dir = "checkpoints"
save_path = "models/"
save_name = 'm1'
EPOCHS = 100
inepoch = 0
max_lr = 0.1

np.random.seed(seeds)
tf.random.set_seed(seeds)

with open(filepath, encoding="UTF-8") as fp:
    text = fp.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index)
print("Total distinct characters: ", vocab_size)
[encoded_text] = np.array(tokenizer.texts_to_sequences([text])) - 1
print("Total characters: ", len(encoded_text))
'''train_ds = tf.data.Dataset.from_tensor_slices(encoded_text)

train_ds = train_ds.window(window_length, shift=n_steps, drop_remainder=True)
train_ds = train_ds.flat_map(lambda window: window.batch(window_length))
train_ds = train_ds.batch(1)
train_ds = train_ds.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
train_ds = train_ds.cache().prefetch(1)'''

window_length = n_steps + 1
encoded_text_parts = np.array_split(encoded_text, batch_size)
train_dss = []
for encoded_text_part in encoded_text_parts:
    train_ds = tf.data.Dataset.from_tensor_slices(encoded_text_part)
    train_ds = train_ds.window(window_length, shift=n_steps, drop_remainder=True)
    train_ds = train_ds.flat_map(lambda window: window.batch(window_length))
    train_dss.append(train_ds)
train_ds = tf.data.Dataset.zip(tuple(train_dss)).map(lambda *windows: tf.stack(windows))
train_ds = train_ds.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
train_ds = train_ds.prefetch(1)
#train_ds = train_ds.cache().prefetch(1)
del train_dss, encoded_text_parts, encoded_text, text

def get_uncompiled_model():
    return keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=(batch_size, None)),
    keras.layers.GRU(rnn_cells, return_sequences=True, stateful=True,
                        dropout=0.1, recurrent_dropout=0.1),
    keras.layers.GRU(rnn_cells, return_sequences=True, stateful=True,
                        dropout=0.1, recurrent_dropout=0.1),
    keras.layers.TimeDistributed(keras.layers.Dense(vocab_size, activation="softmax"))
    ])

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")
    return model

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    global inepoch
    if os.path.exists(checkpoint_dir):
        checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            inepoch = int(latest_checkpoint.split('_')[-1])
            return keras.models.load_model(latest_checkpoint)
        else:
            inepoch = 0
            print("Creating a new model")
            return get_compiled_model()
    else:
        os.makedirs(checkpoint_dir)
        inepoch = 0
        print("Creating a new model")
        return get_compiled_model()

def save_tokenizer(tokenizer, path):
    with open(path + "/tokenizer.json", mode='w') as fp:
        fp.write(tokenizer.to_json())

'''model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=(batch_size, None)),
    keras.layers.GRU(rnn_cells, return_sequences=True, stateful=True,
                        dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(rnn_cells, return_sequences=True, stateful=True,
                        dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(vocab_size, activation="softmax"))
])
#model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=max_lr / 10, momentum=0.95, nesterov=True))
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")'''
model = make_or_restore_model()
print(model.summary())

class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()

class OneCycleLRCallback(keras.callbacks.Callback):
    def __init__(self, max_learning_rate, t_epoch):
        self.lr1 = max_learning_rate
        self.lr0 = self.lr1 / 10
        self.lr2 = self.lr0 / 10
        self.E = t_epoch
        self.b1 = 1 - 0.45 * self.E
        self.b2 = 1 - 0.9 * self.E
        self.k1 = (self.lr0 - self.lr1) / (-self.b1)
        self.k2 = (self.lr2 - self.lr0) / (0.1 * self.E)
        self.k3 = 0.1 / (-self.b1)
        self.det = -self.b2
    def on_epoch_begin(self, epoch, logs=None):
        c_epoch = float(epoch)
        if (c_epoch < self.det):
            self.model.optimizer.learning_rate = self.k1 * abs(c_epoch + self.b1) + self.lr1
            self.model.optimizer.momentum = self.k3 * abs(c_epoch + self.b1) + 0.85
        else:
            self.model.optimizer.learning_rate = self.k2 * (c_epoch + self.b2) + self.lr0
            self.model.optimizer.momentum = 0.95
    def on_epoch_end(self, epoch, logs=None):
        #logs = logs or {}
        logs['lr'] = self.model.optimizer.learning_rate
        logs['momentum'] = self.model.optimizer.momentum

rs_callback = ResetStatesCallback()
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + '/' + save_name + "_{epoch}", save_weights_only=True, save_best_only=True, monitor='loss')
es_callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='loss')
#lr_callback = OneCycleLRCallback(max_lr, EPOCHS)

'''inepoch = int(sys.argv[1])

if inepoch != -1:
    model.load_weights(checkpoint_path)
else:
    inepoch = 0'''

#model.fit(train_ds, epochs=EPOCHS, initial_epoch=inepoch, callbacks=[rs_callback, lr_callback, cp_callback, es_callback])
model.fit(train_ds, epochs=EPOCHS, initial_epoch=inepoch, callbacks=[rs_callback, cp_callback, es_callback])

model.save(save_path + save_name)
save_tokenizer(tokenizer, save_path + save_name)

stateless_model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embed_dim, input_shape=(None,)),
    keras.layers.GRU(rnn_cells, return_sequences=True),
    keras.layers.GRU(rnn_cells, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(vocab_size, activation="softmax"))
])
#stateless_model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=max_lr / 10, momentum=0.95, nesterov=True))
stateless_model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")
stateless_model.set_weights(model.get_weights())
stateless_model.save(save_path + save_name + "_stateless")
save_tokenizer(tokenizer, save_path + save_name + "_stateless")
