# ~/usr/bin/env python
"""
The main script for running experiments
"""
from data import get_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

def main():
    dataset_directory = 'data'
    dataset = 'spam'  # volcanoes #voting #spam
    schema, X, y = get_dataset(dataset, dataset_directory)
    print(len(X))

# Method for training a dataset with Decision Tree Classifier (max_depth = 0 for default)
def train_dtc(dataset_name, test_size, criterion, max_depth):
    # Get dataset
    dataset_directory = 'data'
    dataset = dataset_name  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=1723)

    # Extra model options
    if max_depth == 0:
        clf = DecisionTreeClassifier(criterion=criterion, random_state=1723)
        clf = clf.fit(X_train, y_train)
    else:
        clf = DecisionTreeClassifier(criterion=criterion, random_state=1723, max_depth=max_depth)
        clf = clf.fit(X_train, y_train)
          
    # Model metrics
    score = clf.score(X_test, y_test)
    height = clf.get_depth()
    num_leaves = clf.get_n_leaves()
    
    # Print results
    print(dataset_name, "accuracy with dtc using", criterion, ":", score)
    print(dataset_name, "height:", height)
    print(dataset_name, "number of leaves:", num_leaves)
    
    return [score]


def _1_1_a():
    # Train Decision Tree Classifier with 80/20 ratio an entropy node selection
    train_dtc('spam', 0.2, 'entropy', 0)
    train_dtc('volcanoes', 0.2, 'entropy', 0)
    train_dtc('voting', 0.2, 'entropy', 0)
    

def _1_1_b():
    # Get dataset
    dataset_directory = 'data'
    dataset = 'voting'  
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1723)
    
    # Train Decision Tree Classifier using gini node selection
    clf = DecisionTreeClassifier(criterion='gini', random_state=1723)
    clf = clf.fit(X_train, y_train)
            
    # Plot tree and display it
    tree.plot_tree(clf)
    plt.show()
    
    # Print feature names
    feature = schema.feature_names
    print(feature)    
       
    # Print model prediction accuracy
    score = clf.score(X_test, y_test)    
    print("voting gini accuracy:", score)
    

def _1_1_c():
    # Create numpy arrays for dataset
    graph_x = np.arange(1,51)
    graph_y = np.array([])
    
    # Plot depth as x-axis, accuracy as y-axis
    for i in range(1, 51):
        score = train_dtc('spam', 0.2, 'entropy', i)
        graph_y = np.append(graph_y, score)

    print(graph_y)

    # Print results
    plt.title("Relation Between Tree Depth and Prediction Accuracy")
    plt.xlabel("Node Depth")
    plt.ylabel("Prediction Accuracy")
    plt.plot(graph_x, graph_y, color ="red")
    plt.show()
            
def _1_2():
    # Create Numpy array for dataset
    graph_x = np.array([10, 30, 40, 60])
    graph_y = np.array([])
    
    # Train 90/10, 70/30, 60/40, 40/60 training/testing ratios using entropy node selection
    # Add predition accuracy to array
    score = train_dtc('volcanoes', 0.1, 'entropy', 0)
    graph_y = np.append(graph_y, score)
    
    score = train_dtc('volcanoes', 0.3, 'entropy', 0)
    graph_y = np.append(graph_y, score)

    score = train_dtc('volcanoes', 0.4, 'entropy', 0)
    graph_y = np.append(graph_y, score)
    
    score = train_dtc('volcanoes', 0.6, 'entropy', 0)
    graph_y = np.append(graph_y, score)
    
    # Display results
    plt.title("Relation Between Training and Testing Ratio's")
    plt.xlabel("Training to Testing Ratio")
    plt.ylabel("Prediction Accuracy")
    plt.plot(graph_x, graph_y, color ="red")
    plt.show()


def _2_1_a():   
    # Get dataset
    dataset_directory = 'data'
    dataset = 'voting'
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split data into 80% train and 20% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1723)
    
    # Train Logistic Regression Classifier 
    logReg = LogisticRegression(random_state=1723).fit(X_train, y_train) 
    prob_estimates = logReg.predict_proba(X_train[:2, :])   
    
    # Report accuracy of model
    logReg_accuracy = logReg.score(X_test, y_test)
    
    print("logistic regression prediction accuracy:", logReg_accuracy)    
    print("logistic regression probability estimate:", prob_estimates)    
    
    
def _2_1_b():
    # Get dataset
    dataset_directory = 'data'
    dataset = 'voting'
    schema, X, y = get_dataset(dataset, dataset_directory)
    
    # Split data into 80% train and 20% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1723)
    
    # Train Logistic Regression Classifier 
    logReg = LogisticRegression(random_state=1723).fit(X_train, y_train)    
    
    # Report prediction probability of the logistic regression model
    log_reg_prob = logReg.predict_proba(X_test[:2, :])
    
    # Report prediction probability of the decision tree classifier model
    clf = DecisionTreeClassifier(random_state=1723).fit(X_train, y_train)
    dtc_prob = clf.predict_proba(X_test[:2, :])
   
    # Print results
    print("logistic regression probility prediction:", log_reg_prob)
    print("decision tree probabiliy prediction:", dtc_prob)
    
    
def _3_1():
    # Create dataset
    # Outlook: sunny=1, overcast=2, rain=3 
    # Temp: hot=1, mild=2, cool=3 
    # Humidity: high=1, normal=2 
    # Wind: strong=1, weak=2
    # Play Mins: XX    
    
    # Features
    X = np.array([[1, 1, 1, 2], 
                 [1, 1, 1, 1],
                 [2, 1, 1, 1],
                 [3, 2, 1, 2],
                 [3, 3, 2, 2],
                 [3, 3, 2, 1],
                 [2, 3, 2, 1],
                 [1, 2, 1, 2],
                 [1, 3, 2, 2],
                 [3, 2, 2, 2],
                 [1, 2, 2, 1],
                 [2, 2, 1, 1],
                 [2, 1, 2, 2],
                 [3, 2, 1, 1]])
    
    # Labels  
    y = np.array([25, 30, 48, 46, 62, 23, 43, 36, 38, 48, 48, 62, 44, 30])
    
    # Train Linear Regression model
    linear_regressor = LinearRegression()
    linear_regressor.fit(X,y)
    
    # Create testing dataset
    day_15 = np.array([[3, 1, 1, 2]])    
    # Generate prediction of the testing data using the model
    day_15_prediction = linear_regressor.predict(day_15)
    
    # Get R squared value
    r_squared = linear_regressor.score(X,y)
    
    # Print results
    print("Day 15 predicted value:", day_15_prediction)
    print("R squared:", r_squared)
    
    
def _3_2():
    # Features dataset
    X = np.array([[1, 1, 1, 2], 
                 [1, 1, 1, 1],
                 [2, 1, 1, 1],
                 [3, 2, 1, 2],
                 [3, 3, 2, 2],
                 [3, 3, 2, 1],
                 [2, 3, 2, 1],
                 [1, 2, 1, 2],
                 [1, 3, 2, 2],
                 [3, 2, 2, 2],
                 [1, 2, 2, 1],
                 [2, 2, 1, 1],
                 [2, 1, 2, 2],
                 [3, 2, 1, 1]])
    
    # Labels dataset
    y = np.array([25, 30, 48, 46, 62, 23, 43, 36, 38, 48, 48, 62, 44, 30])
    
    # Train data on decision tree regression model
    model = DecisionTreeRegressor(random_state=1732)
    model.fit(X,y)
    
    # Create testing dataset
    day_15 = np.array([[3, 1, 1, 2]])    
    # Generate prediction of the test dataset using the model
    day_15_prediction = model.predict(day_15)
    
    # Get model information
    num_leaves = model.get_n_leaves()
    height = model.get_depth()
    
    # Print results
    print("Day 15 prediction with regression tree", day_15_prediction)
    print("regression tree number of leaves", num_leaves)
    print("regression tree height", height)
    

if __name__ == "__main__":
    main()
    print("1-1-a:")
    _1_1_a()
    
    print("1-1-b:")
    _1_1_b()
    
    print("1-1-c:")
    _1_1_c()
    
    print("1-2:")
    _1_2()
    
    print("2-1-a:")
    _2_1_a()
    
    print("2-1-b:")
    _2_1_b()
    
    print("3-1")
    _3_1()
    
    print("3-2")
    _3_2()
