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
    # sent, actualVec = enron1.data[0:500], enron1.target[0:500]
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

def lexiconProcessForNN(documents):

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
    X_train, X_test, y_train, y_test = train_test_split(freqVec, acturalVec, test_size=0.3, random_state=5)

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


# The random seed for this NNSplitSets are same with spiltTestSets
# it will generate the different content (one is vector and another is text)
# with same order, therefore, it can sure all the test / train data are the same.
# it has been verified that can produce same result several times.
def NNSplitSets(processedDoc, acturalVec):

    from sklearn.model_selection import train_test_split
    NN_X_train,NN_X_test,NN_y_train,NN_y_test = train_test_split(processedDoc, acturalVec, test_size=0.3, random_state=5)

    return (NN_X_train,NN_X_test,NN_y_train,NN_y_test)


def determineFeature(NN_X_train,NN_X_test,NN_y_train,NN_y_test, lexicon):

    print("--------------- Process with features --------------- ")
    #convet to list
    NN_X_train = list(NN_X_train)
    NN_X_test = list(NN_X_test)
    NN_y_train = list(NN_y_train)
    NN_y_test = list(NN_y_test)


    #shuffle is not needed, because this document uses the resources from the first test_train split.
    #lemmatizer is no longer needed, the sentences has being processed at the pre-processing phase.

    Trainfeatures = []
    TrainfeatureRes = []
    Testfeatures = []
    TestfeatureRes = []

    #process for train sets
    for index in range(len(NN_X_train)):

        if(index % 3000 == 0):
            print("TrainSet progress: ", (index / len(NN_X_train)) * 100, "%")

        #ham
        if(NN_y_train[index] == 0):

            feature = NNsample_Handling(NN_X_train[index], lexicon, [1,0])


        elif (NN_y_train[index] == 1):

            feature = NNsample_Handling(NN_X_train[index], lexicon, [0,1])

        Trainfeatures.append(feature[0])
        TrainfeatureRes.append(feature[1])


    #process for test_sets
    for index in range(len(NN_X_test)):

        if(index % 3000 == 0):
            print("TestSet progress: ", (index / len(NN_X_test)) * 100, "%")

        if(NN_y_test[index] == 0):

            feature = NNsample_Handling(NN_X_test[index], lexicon, [1,0])
        elif(NN_y_test[index] == 1):

            feature = NNsample_Handling(NN_X_test[index], lexicon, [0,1])


        Testfeatures.append(feature[0])
        TestfeatureRes.append(feature[1])

    print("Feature Process Done")
    print("------------   END  -----------------")

    return (list(Trainfeatures), list(TrainfeatureRes), list(Testfeatures), list(TestfeatureRes))


def NNsample_Handling(sents, lexicon, classification):

    import numpy as np

    words = word_tokenize(sents)
    # the word are being processed previously in the pre-processing part
    features = np.zeros(len(lexicon))

    for text in words:
        if text in lexicon:
            index_value = lexicon.index(text)
            features[index_value] += 1

    features = list(features)

    return [features,classification]

def Verification(y_train, y_test, NN_y_train, NN_y_test):

    if (len(y_train) == len(NN_y_train) and len(NN_y_test) == len(y_test)):

        for index in range(len(y_train)):
            if NN_y_train[index][1] != y_train[index]:
                print("------------ ERROR: Train answer did not match")
                return -1


        for index in range(len(y_test)):
            if NN_y_test[index][1] != y_test[index]:
                print("------------ ERROR: Test answer did not match")
                return -1

    else:
        print("------------------- ERROR: size is different for two sets")
        return -1

    print("Size and Content Test Passed...")
    return 1

if __name__ == '__main__':
    (processedDoc, acturalVec) = preprocess()
    (X_train,X_test,y_train,y_test) = spiltTestSets(processedDoc, acturalVec)

    print("X_Train Size: ",len(X_train))
    print("X_Test Size: ",len(X_test))

    # naiveBayes(X_train,X_test,y_train,y_test)
    # randomForest(X_train,X_test,y_train,y_test)


    lexicon = lexiconProcessForNN(processedDoc)
    (NN_X_train,NN_X_test,NN_y_train,NN_y_test) = NNSplitSets(processedDoc, acturalVec)
    (Trainfeatures, TrainfeatureRes, Testfeatures, TestfeatureRes) = determineFeature(NN_X_train,NN_X_test,NN_y_train,NN_y_test,lexicon)
    res = Verification(y_train,y_test,TrainfeatureRes, TestfeatureRes)

    if(res == 1):
        # only continue with passed result

        print("--------- DUMP FILES ---------")

        print("1. X_TRAIN : START")
        with open('pickles/regularAlgorithms_X_TRAIN.pickle','wb') as f:
            pickle.dump(X_train,f)
        print("   X_TRAIN : FINISHED")

        print("2. X_TEST : START")
        with open('pickles/regularAlgorithms_X_TEST.pickle', 'wb') as f:
            pickle.dump(X_test, f)
        print("   X_TEST : FINISHED")

        print("3. Y_TRAIN : START")
        with open('pickles/regularAlgorithms_Y_TRAIN.pickle', 'wb') as f:
            pickle.dump(y_train, f)
        print("   Y_TRAIN : FINISHED")

        print("4. Y_TEST : START")
        with open('pickles/regularAlgorithms_Y_TEST.pickle', 'wb') as f:
            pickle.dump(y_test, f)
        print("   Y_TEST : FINISHED")

        print("5. NN_X_TRAIN : START")
        with open('pickles/NNAlgorithms_X_TRAIN.pickle', 'wb') as f2:
            pickle.dump(Trainfeatures,f2)
        print("   NN_X_TRAIN : FINISHED")

        print("6. NN_X_TEST : START")
        with open('pickles/NNAlgorithms_X_TEST.pickle', 'wb') as f2:
            pickle.dump(Testfeatures,f2)
        print("   NN_X_TEST : FINISHED")

        print("7. NN_Y_TRAIN : START")
        with open('pickles/NNAlgorithms_Y_TRAIN.pickle', 'wb') as f2:
            pickle.dump(TrainfeatureRes,f2)
        print("   NN_Y_TRAIN : FINISHED")

        print("8. NN_Y_TEST : START")
        with open('pickles/NNAlgorithms_Y_TEST.pickle', 'wb') as f2:
            pickle.dump(TestfeatureRes,f2)
        print("   NN_Y_TEST : FINISHED")

        print("--------- END ---------")






