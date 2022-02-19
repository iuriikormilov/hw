import gensim.downloader as api

one = api.info()['models'].keys()
#print(one)

word_vectors = api.load("glove-wiki-gigaword-100")  # загрузим предтренированные вектора слов из gensim-data
# выведим слово наиболее близкое к 'woman', 'king' и далекое от 'man'
result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))


# выведем лишнее слово
print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))

print(word_vectors.doesnt_match("black green summer brown".split()))

# определим схожесть между словами
similarity = word_vectors.similarity('woman', 'man')
print(similarity)

similarity = word_vectors.similarity('human', 'man')
print(similarity)

similarity = word_vectors.similarity('bee', 'man')
print(similarity)



# найдем top-3 самых близких слов
result = word_vectors.similar_by_word("man", topn=3)
print(result)

result = word_vectors.similar_by_word("cat", topn=3)
print(result)

result = word_vectors.similar_by_word("mouth", topn=3)
print(result)