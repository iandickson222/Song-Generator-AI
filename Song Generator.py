#!/usr/bin/env python
# coding: utf-8

## Song Generator

import tensorflow as tf
import numpy as np
import pandas as pd
import re
import random
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
from time import sleep


## Collect Top 100 Songs Data

chart_page = requests.get("https://www.billboard.com/charts/Hot-100")
soup = BeautifulSoup(chart_page.content, 'html.parser')

chart_list_items = soup.find_all('div', class_='chart-list-item')

billboard = pd.DataFrame({
    "position": [item.find(class_='chart-list-item__position').text.replace('\n', '').strip() for item in chart_list_items],
    "title": [item.find(class_='chart-list-item__title').text.replace('\n', '').lower().split(' (', 1)[0].strip() for item in chart_list_items],
    "artist": [item.find(class_='chart-list-item__artist').text.lower().replace('\n', '').split('featuring', 1)[0].split(' x ', 1)[0].split(' & ', 1)[0].split(' + ', 1)[0].strip() for item in chart_list_items]
})

billboard.drop(3, inplace=True)


## Collect Lyrics for Songs

driver = Chrome('Desktop/ChromeDriver.exe')
driver.maximize_window()
driver.get('https://genius.com/')
sleep(5)

lyrics = []
for position, title, artist in billboard.to_numpy():
    search_form = driver.find_element_by_name('q')
    search_form.clear()
    search_form.send_keys(f'{title} by {artist}')
    search_form.submit()
    sleep(5)
    
    try:
        driver.find_elements_by_class_name('mini_card')[1].click()
        sleep(5)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        contents = soup.find('div', class_='lyrics').find('p').contents

        for line in contents:
            line = str(line)
            
            if line.count('<i>') > 0:
                line = ''
            
            if line.count('</i>') > 0:
                line = ''

            if line.count('>') > 0:     
                line = line.split('>', 2)[1]

            if line.count('<') > 0:
                line = line.split('<', 2)[0]

            if line.count('[') > 0:
                line = ''
                
            if line.count(']') > 0:
                line = ''

            if line != '' and line.__repr__().count('\\') == 0:       
                line = line.lower()
                line = re.sub(r'[^\w\s]', '', line)
                line = re.sub(' +', ' ', line)
                
                if line != '':
                    lyrics.append(line)

        sleep(10)
    
    except:
        pass

driver.close()
pd.DataFrame({'lyrics': lyrics}).to_csv('song_lyrics.csv', index = False)


## Tokenize Lyrics

lyrics = pd.read_csv('song_lyrics.csv').values.reshape(-1).tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lyrics)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in lyrics:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(2, len(token_list) + 1):       
        sequence = token_list[:i]
        input_sequences.append(sequence)

max_len = max([len(sequence) for sequence in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_len, padding = 'pre'))
X = input_sequences[:, :-1]
y = input_sequences[:, -1]


## Build Model

model = tf.keras.Sequential([
    keras.layers.Embedding(total_words, 256, input_length = max_len - 1),
    keras.layers.Bidirectional(keras.layers.LSTM(128)),
    keras.layers.Dense(total_words)
])

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              optimizer = keras.optimizers.Adam(lr = 0.01),
              metrics = ['accuracy'])

model.fit(X, y, epochs=10)


## Generate Song

song = []

verses = 25
for v in range(verses):
    
    verse = ''
    
    random_word = random.randint(1, total_words)
    for word, index in tokenizer.word_index.items():
        if index == random_word:
            verse = word
            break
    
    num_words = random.randint(3, 7)               
    for w in range(num_words):
        token_list = tokenizer.texts_to_sequences([verse])[0]
        token_list = pad_sequences([token_list], maxlen = max_len - 1, padding='pre')
        predicted_word = model.predict(token_list)
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted_word):
                verse += ' ' + word
                break
                
    song.append(verse) 

print(song)




