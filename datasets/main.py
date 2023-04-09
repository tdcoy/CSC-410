# ~/usr/bin/env python
"""
The main script for running experiments
"""
from data import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
    # Get dataset
    dataset_directory = 'data'
    dataset = 'voting'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)
    
    clf = MLPClassifier(hidden_layer_sizes=(1,1) ,random_state=seed, max_iter=1000).fit(X_train, y_train)
    onehl_score = clf.score(X_test, y_test)
    print("accuracy for 1 hidden layer: " , onehl_score)
    
    clf = MLPClassifier(hidden_layer_sizes=(1,100) ,random_state=seed, max_iter=1000).fit(X_train, y_train)
    onehund_hl_score = clf.score(X_test, y_test)
    print("accuracy for 100 hidden layers: " , onehund_hl_score)
    
    #Plot Data    
    data = {'1 Hidden Layer':onehl_score, '100 Hidden Layers':onehund_hl_score}
    
    model_metrics = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (4, 3))
    plt.bar(model_metrics, values, color ='blue',
            width = 0.4)
    
    plt.xlabel("Hidden Layers")
    plt.ylabel("Accuracy")
    plt.title("Voting Hidden Layer Accuracy")
    plt.show()
    
def _1_1_b():
    # Get dataset
    dataset_directory = 'data'
    dataset = 'voting'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)
    
    clf = MLPClassifier(hidden_layer_sizes=(1,100), 
                        learning_rate='constant', 
                        learning_rate_init=0.00001, 
                        random_state=seed, max_iter=3000).fit(X_train, y_train)
    
    ln00001_score = clf.score(X_test, y_test)
    print("accuracy for 1 hidden layer: " , ln00001_score)
    
    clf = MLPClassifier(hidden_layer_sizes=(1,100), 
                        learning_rate='constant', 
                        learning_rate_init=0.1, 
                        random_state=seed, max_iter=3000).fit(X_train, y_train)
    
    ln01_score = clf.score(X_test, y_test)
    print("accuracy for 1 hidden layer: " , ln01_score)
    
    #Plot Data    
    data = {'Learning Rate: 0.00001':ln00001_score, 'Learning Rate: 0.1':ln01_score}
    
    model_metrics = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (4, 3))
    plt.bar(model_metrics, values, color ='blue',
            width = 0.4)
    
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Voting Learning Rate Accuracy")
    plt.show()
            
if __name__ == "__main__":
    #main()
    #print("1-1-a:")
    #_1_1_a()
    
    #print("1-1-b:")
    #_1_1_b()