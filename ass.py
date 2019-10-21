import nltk
from nltk import re
from nltk.tokenize import word_tokenize
from sklearn.datasets import load_files
from collections import Counter

nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords


# The data has being process to using one folder to cover all the file, in total there is going to be 33716 emails.
# using shuffle to make sure the data is being read in such random method.

def preprocess():
    from nltk.stem import WordNetLemmatizer

    # load file from doc
    enron1 = load_files(r"email_data/enron1")

    # X will contain the sentences and the target will be their categories.
    sent, actualVec = enron1.data, enron1.target

    # verify the length of the document, it should be 33716.
    print(len(sent))
    print(len(actualVec))

    documents = []

    for index in range(0, len(sent)):

        # remove all special characters
        document = re.sub(r'\W', ' ', str(sent[index]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Removing 'subject' from document
        document = re.sub(r'^Subject\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Lemmatization, to split the sent into words
        document = document.split()

        # change the state of the word e.g. apples -> apple
        stemmer = WordNetLemmatizer()
        document = [stemmer.lemmatize(word) for word in document]

        # combine the processed words back together
        document = ' '.join(document)

        documents.append(document)

        if(index % 3000 == 0):
            print("Pre-Processing Status: ", index/len(sent)*100, "%")

    print("Pre-Processing Finalized")

    return (documents,actualVec)

def futherProcessForNN(documents):

    print("-----------------------------")
    print("Currently process lexicon for Neural Network")
    lexicon = []
    for words in documents:
        wordLists = word_tokenize(words)
        lexicon += list(wordLists)

    passLexicon = []
    word_count = Counter(lexicon)

    for words in word_count:
        if 1000 > word_count[words] > 50:
            passLexicon.append(words)

    print("Lexicon size: ", len(passLexicon))
    print("Lexicon: ")
    print(passLexicon)

    print("NN Pre-Process Finalized")
    print("-----------------------------")

    return passLexicon


def wordToTFIDF(documents):

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    resVec = vectorizer.fit_transform(documents).toarray()
    print(resVec)

    tfidfconverter = TfidfTransformer()
    freqVec = tfidfconverter.fit_transform(resVec).toarray()
    print(freqVec)

    return freqVec

def spiltTestSets(documents, acturalVec):
    freqVec = wordToTFIDF(documents)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(freqVec, acturalVec, test_size=0.3, random_state=0)

    return (X_train,X_test,y_train,y_test)

def naiveBayes(X_train,X_test,y_train,y_test):

    print("\n------------NAIVE BAYES -----------------")

    import datetime
    starttime = datetime.datetime.now()

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()

    #train data
    classifier.fit(X_train, y_train)

    #test data
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("F-Score: ",f1_score(y_test,y_pred))
    print("accuracy: ",accuracy_score(y_test, y_pred))

    endtime = datetime.datetime.now()
    print("Time elapsed: ", endtime - starttime)
    print("------------   END  -----------------")


def randomForest(X_train,X_test,y_train,y_test):

    print("\n------------RANDOM FOREST -----------------")

    import datetime
    starttime = datetime.datetime.now()

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)

    #train data
    classifier.fit(X_train, y_train)

    #test data
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("F-Score: ",f1_score(y_test,y_pred))
    print("accuracy: ",accuracy_score(y_test, y_pred))

    endtime = datetime.datetime.now()
    print("Time elapsed: ", endtime - starttime)
    print("------------   END  -----------------")



# # #
# # # '''Save the model for later use using pickle'''
# # with open('text_classifier', 'wb') as picklefile:
# #     pickle.dump(classifier,picklefile)
# # #
# # # '''Retrieve the model and use it.'''
# # with open('text_classifier', 'rb') as training_model:
# #     model = pickle.load(training_model)
# # y_pred2 = classifier.predict(X_test)
# # #
# # print(confusion_matrix(y_test, y_pred2))
# # print(classification_report(y_test, y_pred2))
# # print(accuracy_score(y_test, y_pred2))
# #
# #
# #

if __name__ == '__main__':
    (processedDoc, acturalVec) = preprocess()
    (X_train,X_test,y_train,y_test) = spiltTestSets(processedDoc, acturalVec)
    naiveBayes(X_train,X_test,y_train,y_test)
    randomForest(X_train,X_test,y_train,y_test)




    # futherProcessForNN(processedDoc)