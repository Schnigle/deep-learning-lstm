import data
import torch
import copy
from gensim.models import Word2Vec

# Text file with "sentences" separated by newlines
sentence_file = "data/goblet_book.txt"
# sentence_file = "data/sentences.txt"
# sentence_file = "data/hp_combined.txt"
output_file = sentence_file + ".w2v"

# define training data
word_data = data.WordData(sentence_file, torch.device("cpu"), 0)
sentences = []
current_sentence = []
for w in word_data.word_data:
	current_sentence.append(w)
	if w == "\n":
		sentences.append(copy.deepcopy(current_sentence))
		current_sentence = []
# print(sentences)
# train model
model = Word2Vec(sentences, min_count=1, compute_loss=True)
print(model.get_latest_training_loss())
model.train(sentences, total_examples=model.corpus_count, epochs=1)
print(model.get_latest_training_loss())
# summarize the loaded model
print(model)
model.save(output_file)
# summarize vocabulary
# words = list(model.wv.vocab)
# print(words)
# # access vector for one word
# print(model.wv['they'])
# print(model.wv['Slytherin'])
# print(model.wv['Harry'])
# print(word)
# print(word.shape)
# print(type(word))
print(model.wv.most_similar(['they']))
print(model.wv.most_similar(['said']))
print(model.wv.most_similar(['and']))
print(model.wv.most_similar(['Harry']))
print(model.wv.most_similar(['Slytherin']))
# print(model.wv.most_similar([word])[0])
# a, b = model.wv.most_similar([word])[0]
# print(a)
# print(model.wv.index2word[0])
# save model
# print(model.)
# load model
# new_model = Word2Vec.load(output_file)
# print(new_model)
