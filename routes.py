from flask import render_template, flash, redirect, url_for, request
from app import app
from forms import LoginForm, PostForm
import re
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.layers import Lambda
from keras.models import Sequential


ROOT = app.root_path + '/'
print(ROOT)

util_folder = ROOT + 'Extras/'
char_to_int = pickle.load(open(util_folder + 'char_to_int.p', 'rb'))
int_to_char = pickle.load(open(util_folder + 'int_to_char.p','rb'))
n_vocab = float(len(int_to_char))
global model
model = load_LSTM(temperature=0.5)
model._make_predict_function()




@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template(app.root_path + '/templates/index.html', title='Home')


@app.route('/books')
def books():
    return render_template('books.html')

@app.route('/medicare')
def medicare():
    return render_template('medicare.html')

@app.route('/CV')
def CV():
    return render_template('CV.html')

@app.route('/Hypermaze')
def Hypermaze():
    return render_template('Hypermaze.html')


@app.route('/hunter', methods=['GET', 'POST'])
def hunter():
    form = PostForm()
    if request.method == 'POST':
        global model, graph

        temperature = request.form.get('temp')
        noChars = request.form.get('noChars')

        if not temperature:
             temperature = 0.5
        else:
            start = time.time()
            temperature = np.float(temperature)
            if temperature < 0:
                temperature = 0.5
            model, graph = load_LSTM(temperature=temperature)
            model._make_predict_function()
            print('time to load model after temp change:', time.time() - start, 'seconds')
        if not noChars:
            noChars = 200
        else:
            noChars = np.int(noChars)
            if noChars < 0 or noChars > 200:
                noChars = 200


        #loading the model if it has not been loaded
        start = time.time()
        if model is None:
            model, graph = load_LSTM(temperature=temperature)
            model._make_predict_function()
        print('time to load model:', time.time() - start, 'seconds')

        if request.form.get('submit') or request.form.get('generate'):
            pattern = request.form.get('post')
            start = time.time()
            original, pattern_num = prepare_input(pattern)
            output = run_model(pattern_num, noChars = noChars)
            print('time to generate text:', time.time() - start, 'seconds')
            flash(original + output)

        return redirect(url_for('hunter'))

    return render_template('hunter.html',
                            title='HunterBot',
                            form=form)


def prepare_input(seed = None):
    if seed:
        seed = seed.lower()
        seed = re.sub(r'[^\x00-\x7f]',r'', seed)
        try:
            if seed[-1]!=' ':
                seed = seed + ' '
            original = seed
            str_len = len(seed)
            words = seed.split(' ')
            if str_len >= 100:
                seed = seed[-100:]
            else: #pad with spaces
                copies = 1 + int(np.floor((100 - str_len) / str_len))
                space_left = 100 - copies * str_len
                word_length = np.cumsum([len(w)+1 for w in words[::-1]])
                last_word = np.max(np.nonzero(word_length < space_left))
                partial_sentence = ' '.join(words[-(last_word + 1):])
                seed = ' ' * (space_left - len(partial_sentence)) \
                        + partial_sentence \
                        + seed * copies
            pattern = [char_to_int[w] for w in seed]
        except:
            original = ''
            seeds = pickle.load(open(util_folder + 'seeds.p', 'rb'))
            start = np.random.randint(0, len(seeds)-1)
            pattern = seeds[start]
    else:
        original = ''
        seeds = pickle.load(open(util_folder + 'seeds.p', 'rb'))
        start = np.random.randint(0, len(seeds)-1)
        pattern = seeds[start]
    return original, np.reshape(pattern, (1,100,1))



def run_model(pattern_num, noChars = 50):
    output = []
    counter = 0
    while counter < noChars:
        x = pattern_num / n_vocab
        with graph.as_default():
            preds = model.predict(x).ravel()
        next_char = np.random.choice(np.arange(len(preds)), 1, p=preds)[0]
        output.append(next_char)
        pattern_num[:,:99,:] = pattern_num[:,1:,:]
        pattern_num[:,-1,:] = next_char
        counter += 1
    return ''.join([int_to_char[w] for w in output])


def load_LSTM(temperature=1.0):
    global model
    json_file = open(ROOT + "Extras/model_struct.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(ROOT + "Extras/model_weights.h5")

    new_model = Sequential()
    new_model.add(model.layers[0])
    new_model.add(model.layers[2])
    new_model.add(Lambda(lambda x : x / temperature))
    new_model.add(model.layers[4])
    model = new_model
    graph = tf.get_default_graph()
    return model, graph
