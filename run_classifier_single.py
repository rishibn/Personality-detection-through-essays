# You need to install scikit-learn:
# sudo pip install scikit-learn
#
#
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn

import sys
import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm  # 16.75
from sklearn.naive_bayes import GaussianNB  # 7.40 accuracy
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier  # 34.79
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier  # 34.3858431645,9.63451306963
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier  # 18.0198935924
from sklearn.neural_network import MLPClassifier
from pandas import DataFrame
import trainBuild_single
import csv
from operator import itemgetter


def showOutput(train_labels, predict,test_data, method):
    label=train_labels[0]
    print(method)
    # print(label)
    # print(predict[0])
    # print(labels)
    # print(test_data[0])
    #

    str_output=str(predict[0])
    list_output=list(str_output.replace('[','').replace(']','').split(','))
    results1 = list(map(int, list_output))
    results = map(bool, results1)
    
    # print(results)
    df1 = {'Personality' : label,'Result(True/False)': results,'Result(0/1)': results1}
    dict_df = DataFrame({key : pd.Series(value) for key, value in df1.items()})
    print(dict_df)







train_data = []
train_labels = []
train_labels_name = []
labels=['anticipation', 'joy', 'negative', 'sadness',
                       'disgust', 'positive', 'anger', 'surprise', 'fear', 'trust'
                       ]
x = trainBuild_single.trainBuild()
x.getValues()
x.getStatusProcessed()


with open('D:/MCA matters/4th sem/minor/clean/train_essay_v2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:

        try:
            train_data.append([float(row[0]),
                               float(row[1]),
                               float(row[2]),
                               float(row[3]),
                               float(row[4]),
                               float(row[5]),
                               float(row[6]),
                               float(row[7]),
                               float(row[8]),
                               float(row[9])
                               ]
                              )
            label = [int(row[10]), int(row[11]), int(row[12]), int(row[13]), int(row[14])]
            # print(str(label))
            train_labels.append(str(label))
            train_labels_name.append(['ext', 'neu', 'agr', 'con', 'opn'])
        except Exception as e:
            print(e)
    test_data = []
    test_label = []
    test_label_name = []

    with open('D:/MCA matters/4th sem/minor/clean/train_single_essay_v2.csv') as csv_file:
        csv_reader_test_data = csv.reader(csv_file, delimiter=',')
        line_count1 = 0
        for row in csv_reader_test_data:

            try:
                test_data.append([
                                  float(row[0]),
                                  float(row[1]),
                                  float(row[2]),
                                  float(row[3]),
                                  float(row[4]),
                                  float(row[5]),
                                  float(row[6]),
                                  float(row[7]),
                                  float(row[8]),
                                  float(row[9])
                ]
                                 )

            except Exception as e:
                print(e)
    # print(train_data)
    # GaussianNB, DecisionTreeClassifier,RandomForestClassifier,
    # AdaBoostClassifier,KNeighborsClassifier 1=30.1758038399,MLPClassifier(alpha=1)
    #-------------------------------------------------
    clf_random_forest = RandomForestClassifier()
    clf_random_forest_fit=clf_random_forest.fit(train_data,train_labels)
    clf_random_forest_predict=clf_random_forest_fit.predict(test_data)
    df = DataFrame(test_data[0], labels)
    print(df)
    showOutput(train_labels_name, clf_random_forest_predict,test_data, 'Random Forest Fit ')
    # ------------------------------------------------
    # clf_GaussianNB = GaussianNB()
    # clf_gaussian= clf_GaussianNB.fit(train_data, train_labels)
    # clf_GaussianNB_predict=clf_GaussianNB.predict(train_data)
    # showOutput(train_labels,clf_GaussianNB_predict, 'GaussianNB ')
    # ------------------------------------------------
    # clf_DecisionTreeClassifier = DecisionTreeClassifier()
    # clf_decision = clf_DecisionTreeClassifier.fit(train_data, train_labels)
    # clf_decision_predict = clf_decision.predict(train_data)
    # showOutput(train_labels, clf_decision_predict, 'DecisionTreeClassifier  ')
    # ------------------------------------------------
    classifier_linear = svm.SVC()
    classifier_linear.fit(train_data, train_labels)
    predict = classifier_linear.predict(test_data)
    # print(predict)
    showOutput(train_labels_name, predict,test_data, 'Linear SVM  ')
    # ------------------------------------------------
    clf_knn=KNeighborsClassifier(1)
    clf_knn_data=clf_knn.fit(train_data,train_labels)
    clf_knn_predict=clf_knn_data.predict(test_data)
    showOutput(train_labels_name,clf_knn_predict,test_data,"KNN")
    # clf_knn_predict_test = clf_knn_data.predict(test_data)
    # print(clf_knn_predict_test)
    # for test_data in clf_knn_predict_test:
    #     inputfile = csv.reader(open('process/BillGates.csv', 'r'))
    #     outputfile = open('placelist.txt', 'w')
    #     i = 0
    #     for row in inputfile:
    #         place = row+int(test_data[0])
    #         print(place)
    #         outputfile.write(place + '\n')
    #         i += 1
