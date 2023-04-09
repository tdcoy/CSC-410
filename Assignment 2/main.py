# ~/usr/bin/env python
"""
The main script for running experiments
"""
from data import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

seed = 1723

def main():
    dataset_directory = 'data'
    dataset = 'volcanoes'  # volcanoes #voting #spam
    schema, X, y = get_dataset(dataset, dataset_directory)
    print(len(X))

def _1_1_a():
    # Get dataset, X=training, y=testing
    dataset_directory = 'data'
    dataset = 'volcanoes'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    clf = MultinomialNB(force_alpha=True)
    clf.fit(X_train,y_train)
    
    results = clf.predict(X_test)
    
    # Prediction Accuracy
    prediction = clf.score(X_test,y_test)
    print("Prediction Accuracy: ", prediction)
    
    # Precision = (True Positives) / (True Positives + False Positives)
    precision = metrics.precision_score(y_test, results, average='binary')
    print("Precision Accuracy: ", precision)
    
    # Recall = (True Positives) / (True Positives + False Negatives)
    recall = metrics.recall_score(y_test, results, average='binary')
    print("Recall: ", recall)
    
    # F1 Score = 2 * [(precision * recall)/(precision + recall)]
    f1score = metrics.f1_score(y_test, results, average='binary')
    print("F1 Score: ", f1score)
    
    # Plot results
    data = {'Prediciton':prediction, 'Precision':precision, 'Recall':recall,
            'F1 Score':f1score}
    model_metrics = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (6, 4))
    plt.bar(model_metrics, values, color ='blue',
            width = 0.5)
    
    plt.xlabel("Model Metrics")
    plt.ylabel("Accuracy")
    plt.title("Volcanoe Model Metrics")
    plt.show()
    
def _1_1_b():
    # Get dataset, X=training, y=testing
    dataset_directory = 'data'
    dataset = 'volcanoes'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    lasso = linear_model.Lasso()
    print(cross_val_score(lasso, X_test, y_test, cv=5))
    
def _1_2_a():
    # Get dataset, X=training, y=testing
    dataset_directory = 'data'
    dataset = 'spam'  
    schema, X, y = get_dataset(dataset, dataset_directory)    

    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Prediction Accuracy
    volcano_gaus_acc = clf.score(X_test, y_test)
    print("Prediction Accuracy: ", volcano_gaus_acc)
    return(volcano_gaus_acc)
    
def _1_2_b():
    # Get dataset, X=training, y=testing
    X, y = load_iris(return_X_y=True)

    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Prediction Accuracy
    iris_gaus_acc = clf.score(X_test, y_test)
    print("Prediction Accuracy: ", iris_gaus_acc)
    return(iris_gaus_acc)
    
def _1_2_c():
    # Plot results
    iris = _1_2_b()
    spam = _1_2_a()
    data = {'Spam':spam, 'Iris':iris}
    model_metrics = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (3, 4))
    plt.bar(model_metrics, values, color ='blue',
            width = 0.5)
    
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.title("Gaussian Model Comparison")
    plt.show()
    
def _2_1_a():
    # Get dataset, X=training, y=testing
    dataset_directory = 'data'
    dataset = 'voting'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    clf = SVC(C=.001)
    clf.fit(X_train, y_train)
    c_001 = clf.score(X_test, y_test)
    print("c .001 acc: ", c_001)
    
    clf = SVC(C=.01)
    clf.fit(X_train, y_train)
    c_01 = clf.score(X_test, y_test)
    print("c .01 acc: ", c_01)
    
    clf = SVC(C=.1)
    clf.fit(X_train, y_train)
    c_1 = clf.score(X_test, y_test)
    print("c .1 acc: ", c_1)
    
    clf = SVC()
    clf.fit(X_train, y_train)
    c_0 = clf.score(X_test, y_test)
    print("c 1 acc: ", c_0)
    
    # Plot results
    data = {'C = .001':c_001, 'C = .01':c_01, 'C = .1':c_1, 'C = 1':c_0}
    model_metrics = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (6, 4))
    plt.bar(model_metrics, values, color ='blue',
            width = 0.5)
    
    plt.xlabel("Penalty Parameter Value")
    plt.ylabel("Accuracy")
    plt.title("Penalty Parameter")
    plt.show()
    
def _2_1_b():
    # Get dataset, X=training, y=testing
    dataset_directory = 'data'
    dataset = 'voting'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    clf_svc = SVC()
    clf_svc.fit(X_train, y_train)
    
    clf_mnb = MultinomialNB()
    clf_mnb.fit(X_train, y_train)
    
    y_score_svc = clf_svc.decision_function(X_test)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

    rfc_disp = metrics.RocCurveDisplay.from_estimator(clf_mnb, X_test, y_test, ax=ax1)
    metrics.RocCurveDisplay.from_predictions(y_test, y_score_svc,ax=ax2)
    
    plt.show()
    
def _2_1_c():
    # Get dataset, X=training, y=testing
    dataset_directory = 'data'
    dataset = 'voting'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    # Preproccess data
    mask = np.array([True, True, False, False, False, False, False, False, False, False, False])
    X_masked = X_train[:, mask]
    
    linear_svc = SVC(kernel='linear')
    linear_svc.fit(X_masked, y_train)
    
    rbf_svc = SVC(kernel='rbf')
    rbf_svc.fit(X_masked, y_train)
    
    linear_disp = DecisionBoundaryDisplay.from_estimator(linear_svc, 
                                                  X_masked, 
                                                  response_method="predict")
    
    linear_disp.ax_.scatter(X[:,0], X[:,1], c=y, s=20,edgecolor="k")
    
    rbf_disp = DecisionBoundaryDisplay.from_estimator(rbf_svc, 
                                                  X_masked, 
                                                  response_method="predict")
    
    rbf_disp.ax_.scatter(X[:,0], X[:,1], c=y, s=20,edgecolor="k")
    
    plt.show()
    
            
if __name__ == "__main__":
    main()
    print("1-1-a:")
    _1_1_a()

    print("1-1-b")
    _1_1_b()

    print("1-2-a")
    _1_2_a()

    print("1-2-b")
    _1_2_b()

    print("1-2-c")
    _1_2_c()

    print("2-1-a")
    _2_1_a()

    print("2-1-b")
    _2_1_b()

    print("2-1-c")
    _2_1_c()