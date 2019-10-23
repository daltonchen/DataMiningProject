# Adopted from https://pythonprogramming.net/train-test-tensorflow-deep-learning-tutorial/
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

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Construct the model by summing the previous inputs and passing it through a nonlinear activation function
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output





# Train the network by calculating the error and adjusting the weights hm_epochs number of times.
def train_neural_network(x):

    import datetime
    starttime = datetime.datetime.now()

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

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


