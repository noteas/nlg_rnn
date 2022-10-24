import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

tf.config.threading.set_inter_op_parallelism_threads(1)

def load_tokenizer(path):
    with open(path + "/tokenizer.json") as fp:
        json_str = fp.read()
    return keras.preprocessing.text.tokenizer_from_json(json_str)

if len(sys.argv) <= 2:
    print("Usage: python use.py temperature length")
    sys.exit(-1)

seeds = 42
#filepath = "data/trumpspeech.txt"
model_path = "models/"
model_name = 'm1'

np.random.seed(seeds)
tf.random.set_seed(seeds)

'''with open(filepath, encoding="UTF-8") as fp:
    text = fp.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])'''

tokenizer = load_tokenizer(model_path + model_name)

model = keras.models.load_model(model_path + model_name)
print(model.summary())

def preprocess(text):
    return np.array(tokenizer.texts_to_sequences([text])) - 1

def next_char(text, temperature=1):
    y_proba = model.predict(preprocess(text))[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]
    #return tokenizer.sequences_to_texts(np.argmax(model.predict(preprocess(text)), axis=-1) + 1)[0][-1]
    

def complete_text(text, n_chars=100, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

t = float(sys.argv[1])
l = int(sys.argv[2])
while True:
    s = input("请输入开头：")
    print(complete_text(s, temperature=t, n_chars=l))
