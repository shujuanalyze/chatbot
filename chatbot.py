import numpy as np
import pickle
import operator
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Dropout, Concatenate, RepeatVector, \
    Activation, Dot
from keras.layers import concatenate, dot
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import TruncatedNormal
import pydot
import os, re
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jieba
import requests
import tensorflow as tf
from keras import backend as K


class Chatbot:
    def __init__(self):

        K.clear_session()
        self.graph = tf.compat.v1.get_default_graph()
        self.word_to_index = None
        self.index_to_word = None
        self.question_model = None
        self.answer_model = None

        self.load_data()
        self.build_models()

    def load_data(self):
        question = np.load('pad_question.npy')
        answer = np.load('pad_answer.npy')
        answer_o = np.load('answer_o.npy', allow_pickle=True)
        with open('vocab_bag.pkl', 'rb') as f:
            self.words = pickle.load(f)
        with open('pad_word_to_index.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)
        with open('pad_index_to_word.pkl', 'rb') as f:
            self.index_to_word = pickle.load(f)

        self.vocab_size = len(self.word_to_index) + 1
        self.maxLen = 20

        self.question = question
        self.answer = answer
        self.answer_o = answer_o

    def build_models(self):
        graph = tf.compat.v1.get_default_graph()
        with self.graph.as_default():
            truncatednormal = TruncatedNormal(mean=0.0, stddev=0.05)
            embed_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=100,
                mask_zero=True,
                input_length=None,
                embeddings_initializer=truncatednormal
            )
            LSTM_encoder = LSTM(
                512,
                return_sequences=True,
                return_state=True,
                kernel_initializer='lecun_uniform',
                name='encoder_lstm'
            )
            LSTM_decoder = LSTM(
                512,
                return_sequences=True,
                return_state=True,
                kernel_initializer='lecun_uniform',
                name='decoder_lstm'
            )

            input_question = Input(shape=(None,), dtype='int32', name='input_question')
            input_answer = Input(shape=(None,), dtype='int32', name='input_answer')

            input_question_embed = embed_layer(input_question)
            input_answer_embed = embed_layer(input_answer)

            encoder_lstm, question_h, question_c = LSTM_encoder(input_question_embed)

            decoder_lstm, _, _ = LSTM_decoder(input_answer_embed, initial_state=[question_h, question_c])

            attention = dot([decoder_lstm, encoder_lstm], axes=[2, 2])
            attention = Activation('softmax')(attention)
            context = dot([attention, encoder_lstm], axes=[2, 1])
            decoder_combined_context = concatenate([context, decoder_lstm])

            decoder_dense1 = TimeDistributed(Dense(256, activation="tanh"))
            decoder_dense2 = TimeDistributed(Dense(self.vocab_size, activation="softmax"))
            output = decoder_dense1(decoder_combined_context)
            output = decoder_dense2(output)

            with graph.as_default():  # 添加这行代码
                model = Model([input_question, input_answer], output)
                model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

                model.load_weights('models/W--184-0.5949-.h5')
                self.model = model

            question_model = Model(input_question, [encoder_lstm, question_h, question_c])
            self.question_model = question_model

            answer_h = Input(shape=(512,))
            answer_c = Input(shape=(512,))
            encoder_lstm = Input(shape=(self.maxLen, 512))
            target, h, c = LSTM_decoder(input_answer_embed, initial_state=[answer_h, answer_c])
            attention = dot([target, encoder_lstm], axes=[2, 2])
            attention_ = Activation('softmax')(attention)
            context = dot([attention_, encoder_lstm], axes=[2, 1])
            decoder_combined_context = concatenate([context, target])
            output = decoder_dense1(decoder_combined_context)
            output = decoder_dense2(output)
            answer_model = Model([input_answer, answer_h, answer_c, encoder_lstm], [output, h, c, attention_])
            self.answer_model = answer_model

    def act_weather(self, city):
        url = 'http://wthrcdn.etouch.cn/weather_mini?city=' + city
        page = requests.get(url)
        data = page.json()
        temperature = data['data']['wendu']
        notice = data['data']['ganmao']
        outstrs = "地点： %s\n气温： %s\n注意： %s" % (city, temperature, notice)
        return outstrs + ' EOS'

    def input_question(self, seq):
        seq = jieba.lcut(seq.strip(), cut_all=False)
        sentence = seq
        try:
            seq = np.array([self.word_to_index[w] for w in seq])
        except KeyError:
            seq = np.array([36874, 165, 14625])
        seq = sequence.pad_sequences([seq], maxlen=self.maxLen, padding='post', truncating='post')
        return seq, sentence

    def decode_greedy(self, seq, sentence):
        question = seq
        for index in question[0]:
            if int(index) == 5900:
                for index_ in question[0]:
                    if index_ in [7851, 11842, 2406, 3485, 823, 12773, 8078]:
                        return self.act_weather(self.index_to_word[index_])
        answer = np.zeros((1, 1))
        attention_plot = np.zeros((20, 20))
        answer[0, 0] = self.word_to_index['BOS']
        i = 1
        answer_ = []
        flag = 0
        encoder_lstm_, question_h, question_c = self.question_model.predict(x=question, verbose=1)
        while flag != 1:
            prediction, prediction_h, prediction_c, attention = self.answer_model.predict([
                answer, question_h, question_c, encoder_lstm_
            ])
            attention_weights = attention.reshape(-1, )
            attention_plot[i] = attention_weights
            word_arg = np.argmax(prediction[0, -1, :])
            answer_.append(self.index_to_word[word_arg])
            if word_arg == self.word_to_index['EOS'] or i > 20:
                flag = 1
            answer = np.zeros((1, 1))
            answer[0, 0] = word_arg
            question_h = prediction_h
            question_c = prediction_c
            i += 1
        result = ' '.join(answer_)
        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence)]
        self.plot_attention(attention_plot, sentence, result.split(' '))
        return ' '.join(answer_)

    def decode_beamsearch(self, seq, beam_size):
        question = seq
        encoder_lstm_, question_h, question_c = self.question_model.predict(x=question, verbose=1)
        sequences = [[[self.word_to_index['BOS']], 1.0, question_h, question_c]]
        answer = np.zeros((1, 1))
        answer[0, 0] = self.word_to_index['BOS']
        answer_ = ''
        flag = 0
        last_words = [self.word_to_index['BOS']]
        for i in range(self.maxLen):
            all_candidates = []
            for j in range(len(sequences)):
                s, score, h, c = sequences[j]
                last_word = s[-1]
                if not isinstance(last_word, int):
                    last_word = last_word[-1]
                answer[0, 0] = last_word
                output, h, c, _ = self.answer_model.predict([answer, h, c, encoder_lstm_])
                output = output[0, -1]
                for k in range(len(output)):
                    candidate = [seq + [k], score * -np.log(output[k]), h, c]
                all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_size]
        answer_ = sequences[0][0]
        print(answer_[0])
        answer_ = [self.index_to_word[x] for x in answer_[0] if (x != 0)]
        answer_ = ' '.join(answer_)
        return answer_

    def plot_attention(self, attention, sentence, predicted_sentence):
        zhfont = matplotlib.font_manager.FontProperties(fname='simkai.ttf')
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        attention = [x[::-1] for x in attention]
        ax.matshow(attention, cmap='viridis')
        fontdict = {'fontsize': 20}
        ax.set_xticklabels([''] + sentence, fontdict=fontdict, fontproperties=zhfont)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict, fontproperties=zhfont)
        plt.show()

    def chat_response(self, input):
        with self.graph.as_default():
           seq, sentence = self.input_question(input)
           answer = self.decode_greedy(seq, sentence)
           return (answer)
