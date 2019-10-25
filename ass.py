import nltk
from nltk import re
from nltk.tokenize import word_tokenize
from sklearn.datasets import load_files
from collections import Counter

nltk.download('stopwords')
import pickle
# import _pickle as cPickle
from nltk.corpus import stopwords

# if reading flag = 0, regular reading (all)
# if reading flag = 1, separate reading (1,3,5 2,4)
readingFlag = 1;# Adopted from https://pythonprogramming.net/train-test-tensorflow-deep-learning-tutorial/
# from preProcessingText import create_feature_sets_and_labels
import tensorflow as tf
import pickle
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# train_x, train_y, test_x, test_y = create_feature_sets_and_labels('data/sentiment2/pos.txt', 'data/sentiment2/neg.txt')


from sklearn.externals import joblib

# pickle_in = open('pickles/test.pickle','rb')
# (train_x, train_y, test_x, test_y) = joblib.load('pickles/joblibtest.pkl')
train_x = joblib.load('pickles/NN_X_train.pkl')
train_y = joblib.load('pickles/NN_Y_train.pkl')
test_x = joblib.load('pickles/NN_X_Test.pkl')
test_y = joblib.load('pickles/NN_Y_Test.pkl')

print(len(train_x))
print(len(train_y))
print(len(test_x))
print(len(test_y))

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
n_nodes_hl4 = 1500


n_classes = 2
batch_size = 100
hm_epochs = 5

x = tf.placeholder('float')
y = tf.placeholder('float')
# Construct the NN by creating individual layers. The first hidden layer is the input layer.
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

hidden_4_layer = {'f_fum': n_nodes_hl4,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl4]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Construct the model by summing the previous inputs and passing it through a nonlinear activation function
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weight']), hidden_4_layer['bias'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4, output_layer['weight']) + output_layer['bias']

    return output





# Train the network by calculating the error and adjusting the weights hm_epochs number of times.
def train_neural_network(x):

    import datetime
    starttime = datetime.datetime.now()

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        #print out accuracy
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: test_x, y: test_y}))

        print("------------------ Evaluation ---------------------")
        print("Perfoming test...")
        pred = []

        count = 0;
        for data in test_x:

            if(count % 3000 == 0):
                print("Current Progress: ", count / len(test_x) * 100, "%")

            result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [data]}), 1)))
            pred.append(result[0])
            count += 1

        print("Extract Actual Value...")
        actural = []
        for element in test_y:
            actural.append(element[1])

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
        print(confusion_matrix(actural,pred))
        print(classification_report(actural,pred))
        print("F-Score: ",f1_score(actural,pred,average='weighted'))
        print("accuracy: ",accuracy_score(actural, pred))

        endtime = datetime.datetime.now()
        print("Time elapsed: ", endtime - starttime)
        print("------------------ END ---------------------")
        print("Please verify two accuracy to see if they matches each other.")

train_neural_network(x)




# The data has being process to using one folder to cover all the file, in total there is going to be 33716 emails.
# using shuffle to make sure the data is being read in such random method.

def loadFiles():

    # load file from doc
    enron1 = load_files(r"email_data/enron1")
    # regular load
    # X will contain the sentences and the target will be their categories.
    sent, actualVec = enron1.data[0:15000], enron1.target[0:15000]
    # sent, actualVec = enron1.data, enron1.target

    # verify the length of the document, it should be 15000.
    print(len(sent))
    print(len(actualVec))

    return (sent, actualVec)



def loadFiles_135():

    # load file from 2 docs
    train = load_files(r"email_data_2/train")
    test = load_files(r"email_data_2/test")

    train_sents, train_actural_vec = train.data[0:10500], train.target[0:10500]
    test_sents, test_actural_vec = test.data[0:4500], test.target[0:4500]

    print(len(train_sents))
    print(len(train_actural_vec))
    print(len(test_sents))
    print(len(test_actural_vec))

    return (train_sents, train_actural_vec, test_sents, test_actural_vec)



def preprocess(sent, actualVec):

    from nltk.stem import WordNetLemmatizer

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

    if readingFlag == 0:
        (sent, actualVec) = loadFiles()

        #pre-all: (train)
        (processedDoc, acturalVec) = preprocess(sent, actualVec)
        (X_train,X_test,y_train,y_test) = spiltTestSets(processedDoc, acturalVec)

        lexicon = lexiconProcessForNN(processedDoc)
        (NN_X_train,NN_X_test,NN_y_train,NN_y_test) = NNSplitSets(processedDoc, acturalVec)



    elif readingFlag == 1:
        (train_sents, train_actural_vec, test_sents, test_actural_vec) = loadFiles_135()

        #pre-process: (train)
        print("For Train sets")
        (train_processed_Doc, train_acturalVec) = preprocess(train_sents, train_actural_vec)

        print("For Test sets")
        #pre-process: (train)
        (test_processed_Doc, test_acturalVec) = preprocess(test_sents, test_actural_vec)

        train_freqVec = wordToTFIDF(train_processed_Doc)
        test_freqVec = wordToTFIDF(test_processed_Doc)

        (X_train,X_test,y_train,y_test) = (train_freqVec, test_freqVec, train_acturalVec, test_acturalVec)


        print("X_Train Size: ",len(X_train))
        print("X_Test Size: ",len(X_test))

        processedDoc = train_processed_Doc + test_processed_Doc
        lexicon = lexiconProcessForNN(processedDoc)

        (NN_X_train,NN_X_test,NN_y_train,NN_y_test) = (train_processed_Doc, test_processed_Doc, train_actural_vec, test_actural_vec)
        print(NN_X_train[0])


    (Trainfeatures, TrainfeatureRes, Testfeatures, TestfeatureRes) = determineFeature(NN_X_train,NN_X_test,NN_y_train,NN_y_test,lexicon)
    res = Verification(y_train,y_test,TrainfeatureRes, TestfeatureRes)

    if(res == 1):
        # only continue with passed result

        print("--------- DUMP FILES ---------")
        from sklearn.externals import joblib

        print("1. X_TRAIN : START")
        joblib.dump(X_train, 'pickles/regularAlgorithms_X_TRAIN.pkl')
        print("   X_TRAIN : FINISHED")

        print("2. X_TEST : START")
        joblib.dump(X_test, 'pickles/regularAlgorithms_X_TEST.pkl')
        print("   X_TEST : FINISHED")

        print("3. Y_TRAIN : START")
        joblib.dump(y_train, 'pickles/regularAlgorithms_Y_TRAIN.pkl')
        print("   Y_TRAIN : FINISHED")

        print("4. Y_TEST : START")
        joblib.dump(y_test, 'pickles/regularAlgorithms_Y_TEST.pkl')
        print("   Y_TEST : FINISHED")



        print("5. NN_X_TRAIN : START")
        joblib.dump(Trainfeatures, 'pickles/NN_X_train.pkl')
        print("   NN_X_TRAIN : FINISHED")


        print("6. NN_Y_TRAIN : START")
        joblib.dump(TrainfeatureRes, 'pickles/NN_Y_train.pkl')
        print("   NN_Y_TRAIN : FINISHED")


        print("7. NN_X_TEST : START")
        joblib.dump(Testfeatures, 'pickles/NN_X_Test.pkl')
        print("   NN_X_TEST : FINISHED")

        print("8. NN_Y_TEST : START")
        joblib.dump(TestfeatureRes, 'pickles/NN_Y_Test.pkl')
        print("   NN_Y_TEST : FINISHED")


        print("--------- Pre-Processing Finished ---------")
        print("Please use normalAlgorithms.py and NeuralNetwork.py to perform test")
        print("Those two .py document need to under the same folder")

















