#https://pythonprogramming.net/preprocessing-tensorflow-deep-learning-tutorial/
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000
#This method creates a lexicon from all the words in the corpus.
#It only uses words which occur more than 50 times and less than 1000 times. Effectively it ignores very frequent (such as a, an to from, etc) and very infrequent words.
def create_lexicon(pos,neg):

	lexicon = []
	with open(pos,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			print("tokenize")
			print(all_words)
			lexicon += list(all_words)
			print(list(all_words))

	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		#print(w_counts[w])
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	print("Lexicon size: ", len(l2))
	print ("Lexicon: ")
	print(l2)
	return l2

#processes the text for tokenisation and lemmatization
def sample_handling(sample,lexicon,classification):

	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features,classification])

	return featureset

#Splits the sample into traning and text sets
def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('neural/Ham.txt',lexicon,[1,0])
	features += sample_handling('neural/Spam.txt',lexicon,[0,1])
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))
#create train and test sets.
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

#creates the train and the test sets and serielizes the data and stores it in a binary file
if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('neural/Ham.txt','neural/Spam.txt')
	# if you want to pickle this data:
	with open('neural/result.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
