import pandas as pd
import numpy as np
import re
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')


apostrophe_dict = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

short_word_dict = {
"121": "one to one",
"a/s/l": "age, sex, location",
"adn": "any day now",
"afaik": "as far as I know",
"afk": "away from keyboard",
"aight": "alright",
"alol": "actually laughing out loud",
"b4": "before",
"b4n": "bye for now",
"bak": "back at the keyboard",
"bf": "boyfriend",
"bff": "best friends forever",
"bfn": "bye for now",
"bg": "big grin",
"bta": "but then again",
"btw": "by the way",
"cid": "crying in disgrace",
"cnp": "continued in my next post",
"cp": "chat post",
"cu": "see you",
"cul": "see you later",
"cul8r": "see you later",
"cya": "bye",
"cyo": "see you online",
"dbau": "doing business as usual",
"fud": "fear, uncertainty, and doubt",
"fwiw": "for what it's worth",
"fyi": "for your information",
"g": "grin",
"g2g": "got to go",
"ga": "go ahead",
"gal": "get a life",
"gf": "girlfriend",
"gfn": "gone for now",
"gmbo": "giggling my butt off",
"gmta": "great minds think alike",
"h8": "hate",
"hagn": "have a good night",
"hdop": "help delete online predators",
"hhis": "hanging head in shame",
"iac": "in any case",
"ianal": "I am not a lawyer",
"ic": "I see",
"idk": "I don't know",
"imao": "in my arrogant opinion",
"imnsho": "in my not so humble opinion",
"imo": "in my opinion",
"iow": "in other words",
"ipn": "I’m posting naked",
"irl": "in real life",
"jk": "just kidding",
"l8r": "later",
"ld": "later, dude",
"ldr": "long distance relationship",
"llta": "lots and lots of thunderous applause",
"lmao": "laugh my ass off",
"lmirl": "let's meet in real life",
"lol": "laugh out loud",
"ltr": "longterm relationship",
"lulab": "love you like a brother",
"lulas": "love you like a sister",
"luv": "love",
"m/f": "male or female",
"m8": "mate",
"milf": "mother I would like to fuck",
"oll": "online love",
"omg": "oh my god",
"otoh": "on the other hand",
"pir": "parent in room",
"ppl": "people",
"r": "are",
"rofl": "roll on the floor laughing",
"rpg": "role playing games",
"ru": "are you",
"shid": "slaps head in disgust",
"somy": "sick of me yet",
"sot": "short of time",
"thanx": "thanks",
"thx": "thanks",
"ttyl": "talk to you later",
"u": "you",
"ur": "you are",
"uw": "you’re welcome",
"wb": "welcome back",
"wfm": "works for me",
"wibni": "wouldn't it be nice if",
"wtf": "what the fuck",
"wtg": "way to go",
"wtgp": "want to go private",
"ym": "young man",
"gr8": "great"
}


emoticon_dict = {
":)": "happy",
":‑)": "happy",
":-]": "happy",
":-3": "happy",
":->": "happy",
"8-)": "happy",
":-}": "happy",
":o)": "happy",
":c)": "happy",
":^)": "happy",
"=]": "happy",
"=)": "happy",
"<3": "happy",
":-(": "sad",
":(": "sad",
":c": "sad",
":<": "sad",
":[": "sad",
">:[": "sad",
":{": "sad",
">:(": "sad",
":-c": "sad",
":-< ": "sad",
":-[": "sad",
":-||": "sad"
}

train_df = pd.read_csv('train_tweets.csv')
print(train_df.head())

test_df = pd.read_csv('test_tweets.csv')
print(test_df.head())

combo = train_df.append(test_df, ignore_index=True, sort=False)
print(combo.head())
print(combo.info())

#2 удали @user
#pat = re.sub(r'@\W', ' ', data)
def without_user(data):
    one = re.sub(r'@user', '', f'{data}')
    return one


new = []
for row in range(len(combo)):
    res = without_user(combo.tweet[row])
    new.append(res)
#print(new)


#3

def lower (data):
    one = data.lower()
    return one


new_one = []
for row in range(len(combo)):
    res = lower(new[row])
    new_one.append(res)
print(new_one[:5])

#4

new_two = []
for part in new_one:
    word = part.split()
    res_one = []
    for w in word:

        if w in apostrophe_dict.keys():
            w = apostrophe_dict[w]
        res_one.append(w)
    res_two = ' '.join([one for one in res_one])
    new_two.append(res_two)

print(new_two[:10])

#5

new_three = []
for part in new_two:
    word = part.split()
    res_one = []
    for w in word:

        if w in short_word_dict.keys():
            w = short_word_dict[w]
        res_one.append(w)
    res_two = ' '.join([one for one in res_one])
    new_three.append(res_two)

print(new_three[:10])

#6

new_four = []

for part in new_three:
    word = part.split()
    res_one = []
    for w in word:

        if w in emoticon_dict.keys():
            w = emoticon_dict[w]
        res_one.append(w)
    res_two = ' '.join([one for one in res_one])
    new_four.append(res_two)

print(new_four[:10])


#7

def change (data):
    one = re.sub(r'[^\w\s]', '', f'{data}')
    return one

new_five = []
for raw in new_four:
    res = change(raw)
    new_five.append(res)

print(new_five[:10])


#8

def simbols (data):
    one = re.sub(r'[^a-zA-Z0-9]', '', f'{data}')
    return one

new_six = []
for raw in new_four:
    res = change(raw)
    new_six.append(res)
print(new_six[:10])



#9

def digits_to_space (data):
    one = re.sub(r'[^a-zA-Z]', ' ', f'{data}')
    return one

new_seven = []

for raw in new_six:
    res = digits_to_space(raw)
    new_seven.append(res)

print(new_seven[:10])

#10

new_eight = []
for part in new_seven:
    word = part.split()
    res_one = []
    for w in word:

        if len(w) > 1:

            res_one.append(w)
    res_two = ' '.join([one for one in res_one])
    new_eight.append(res_two)

print(new_eight[:10])

#11

new_nine = []

for part in new_eight:
    token = nltk.tokenize.word_tokenize(part)
    new_nine.append(token)

print(new_nine[:10])

#13

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

new_ten = []
for part in new_nine:
    sborka = []
    for one in part:
        res = stemmer.stem(one)

        sborka.append(res)
    new_ten.append(sborka)
print(new_ten[:10])

#14
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

new_eleven = []

for parts in new_nine:
    sobr = []
    for one in parts:
        ress = lemmatizer.lemmatize(one)
        sobr.append(ress)
    new_eleven.append(sobr)

print(new_eleven[:10])



