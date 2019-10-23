from sklearn.externals import joblib

train_x = joblib.load('pickles/regularAlgorithms_X_TRAIN.pkl')
train_y = joblib.load('pickles/regularAlgorithms_Y_TRAIN.pkl')
test_x = joblib.load('pickles/regularAlgorithms_X_TEST.pkl')
test_y = joblib.load('pickles/regularAlgorithms_Y_TEST.pkl')

print("Train Set Size:",len(train_x))
print("Train Res Size:",len(train_y))
print("Test Set Size:",len(test_x))
print("Test Res Size:",len(test_y))

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
    print("F-Score: ",f1_score(y_test,y_pred,average='weighted'))
    print("accuracy: ",accuracy_score(y_test, y_pred))

    endtime = datetime.datetime.now()
    print("Time elapsed: ", endtime - starttime)
    print("------------   END  -----------------")

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
    print("F-Score: ",f1_score(y_test,y_pred,average='weighted'))
    print("accuracy: ",accuracy_score(y_test, y_pred))

    endtime = datetime.datetime.now()
    print("Time elapsed: ", endtime - starttime)
    print("------------   END  -----------------")


if __name__ == '__main__':
    naiveBayes(train_x,test_x,train_y,test_y)
    randomForest(train_x,test_x,train_y,test_y)