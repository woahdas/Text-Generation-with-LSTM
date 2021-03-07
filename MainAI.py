import keras
import numpy as np

text = open("Les.txt", encoding="utf8").read().lower()
print('Corpus length:', len(text))

maxlen = 60 #extract sequences of 60 characters

step = 3 #sample a new sequence every three characters

sentences = [] #holds the extracted sequences

next_chars = [] #holds the targets (follow-up characters)

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text))) #list of unique characters in the corpus
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars) #dictionary that maps unique characters to their index in the list "chars"

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)   #one-hot encodes the characters into binary arrays
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)           #
for i, sentence in enumerate(sentences):                            #
    for t, char in enumerate(sentence):                             #
        x[i, t, char_indices[char]] = 1                             #
    y[i, char_indices[next_chars[i]]] = 1                           #

#BUILDING THE NETWORK
from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

#MODEL COMPILATION CONFIGURATION
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#FUNCTION TO SAMPLE THE NEXT CHARACTER GIVEN THE MODEL'S PREDICTIONS
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

import random
import sys

for epoch in range(1, 60): #trains model for 60 epochs
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1) #fits the model for one iteration on the data
    start_index = random.randint(0, len(text) - maxlen - 1)         #
    generated_text = text[start_index: start_index + maxlen]        #selects a text seed at random
    print('--- Generating with seed: "' + generated_text + '"')     #

for temperature in [0.2, 0.5, 1.0, 1.2]: #tries a range of different sampling temperatures
    print('------ temperature:', temperature)
    sys.stdout.write(generated_text)

for i in range(400): #generates 400 characters, starting from seed text
    sampled = np.zeros((1, maxlen, len(chars)))     #
    for t, char in enumerate(generated_text):       #one-hot encodes the characters generated so far
        sampled[0, t, char_indices[char]] = 1.      #

preds = model.predict(sampled, verbose=0)[0]    #samples the next character
next_index = sample(preds, temperature)         #
next_char = chars[next_index]                   #

generated_text += next_char
generated_text = generated_text[1:]

sys.stdout.write(next_char)
