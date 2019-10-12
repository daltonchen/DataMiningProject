# from https://stackabuse.com/text-classification-with-python-and-scikit-learn/
# Text classification and Evaluation
import re
import nltk
import numpy
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords


# The data has being process to using one folder to cover all the file, in total there is going to be 33716 emails.
# using shuffle to make sure the data is being read in such random method.
enron1 = load_files(r"email_data/enron1",shuffle = True)

X, y = enron1.data, enron1.target

print(len(X))

print(X)

documents = []

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    from nltk.stem import WordNetLemmatizer

    stemmer = WordNetLemmatizer()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

print(document)
#
# #Convert the word to a vector using BOW model.
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X = vectorizer.fit_transform(documents).toarray()
#
# '''Using TFIDF instead of BOW, TFIDF also takes into account the frequency instead of just the occurance.
# calculated as:
# Term frequency = (Number of Occurrences of a word)/(Total words in the document)
# IDF(word) = Log((Total number of documents)/(Number of documents containing the word))
# TF-IDF is the product of the two.
# '''
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidfconverter = TfidfTransformer()
# X = tfidfconverter.fit_transform(X).toarray()
#
#
#
# ''' Creating training and test sets of the data'''
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# '''train a clasifier with the data'''
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
# classifier.fit(X_train, y_train)
#
# '''Now predict on the testing data'''
# y_pred = classifier.predict(X_test)
#
# '''Print the evaluation metrices'''

# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))
#
# '''Save the model for later use using pickle'''
# with open('text_classifier', 'wb') as picklefile:
#     pickle.dump(classifier,picklefile)
#
# '''Retrieve the model and use it.'''
# with open('text_classifier', 'rb') as training_model:
#     model = pickle.load(training_model)
# y_pred2 = model.predict(X_test)
#
# print(confusion_matrix(y_test, y_pred2))
# print(classification_report(y_test, y_pred2))
# print(accuracy_score(y_test, y_pred2))



print('END')