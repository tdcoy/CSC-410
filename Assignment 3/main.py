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
    # Get dataset, X=training, y=testing
    dataset_directory = 'data'
    dataset = 'voting'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split the dataset into training and testing subsets randomly by the ratio 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)
    
    clf = MLPClassifier(random_state=seed, max_iter=300).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
            
if __name__ == "__main__":
    main()
    print("1-1-a:")
    _1_1_a()