import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re


#1

new_list = []
with open('tweet_stemmed.csv', encoding='utf-8') as r_file:
    # Создаем объект reader, указываем символ-разделитель ","
    file_reader = csv.reader(r_file, delimiter=";")
    for row in file_reader:
       new_list.append(row)

#print(new_list)

def change (data):
    one = re.sub(r'[^\w\s]', ' ', f'{data}')
    return one

new_list_two = []

for row in new_list:
    one = change(row)
    new_list_two.append(one)
#for row in new_list_two:
    #print(row)


count_vectorizer = CountVectorizer(max_df=0.9, max_features=1000)

bag_of_words = count_vectorizer.fit_transform(new_list_two)
feature_names = count_vectorizer.get_feature_names()

df = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

print(df.head())

new_list = []
with open('tweet_lemmatized.csv', encoding='utf-8') as r_file:
    # Создаем объект reader, указываем символ-разделитель ","
    file_reader = csv.reader(r_file, delimiter=";")
    for row in file_reader:
       new_list.append(row)

#print(new_list)

def change (data):
    one = re.sub(r'[^\w\s]', ' ', f'{data}')
    return one

new_list_two = []

for row in new_list:
    one = change(row)
    new_list_two.append(one)
#for row in new_list_two:
    #print(row)


count_vectorizer = CountVectorizer(max_df=0.9, max_features=1000)

bag_of_words = count_vectorizer.fit_transform(new_list_two)
feature_names = count_vectorizer.get_feature_names()

df = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

print(df.head())



#2
new_list = []
with open('tweet_stemmed.csv', encoding='utf-8') as r_file:
    # Создаем объект reader, указываем символ-разделитель ","
    file_reader = csv.reader(r_file, delimiter=";")
    for row in file_reader:
       new_list.append(row)

#print(new_list)

def change (data):
    one = re.sub(r'[^\w\s]', ' ', f'{data}')
    return one

new_list_two = []

for row in new_list:
    one = change(row)
    new_list_two.append(one)
#for row in new_list_two:
    #print(row)


tf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=1000)

bag_of_words = tf_vectorizer.fit_transform(new_list_two)
feature_names = tf_vectorizer.get_feature_names()

df = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

print(df.head())



new_list = []
with open('tweet_lemmatized.csv', encoding='utf-8') as r_file:
    # Создаем объект reader, указываем символ-разделитель ","
    file_reader = csv.reader(r_file, delimiter=";")
    for row in file_reader:
       new_list.append(row)

#print(new_list)

def change (data):
    one = re.sub(r'[^\w\s]', ' ', f'{data}')
    return one

new_list_two = []

for row in new_list:
    one = change(row)
    new_list_two.append(one)
#for row in new_list_two:
    #print(row)


tf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=1000)

bag_of_words = tf_vectorizer.fit_transform(new_list_two)
feature_names = tf_vectorizer.get_feature_names()

df = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

print(df.head())