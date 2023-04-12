# ~/usr/bin/env python
"""
The main script for running experiments
"""
from data import get_dataset
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

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

    clf = MLPClassifier(hidden_layer_sizes=(
        1, 1), random_state=seed, max_iter=1000).fit(X_train, y_train)
    onehl_score = clf.score(X_test, y_test)
    print("accuracy for 1 hidden layer: ", onehl_score)

    clf = MLPClassifier(hidden_layer_sizes=(1, 100),
                        random_state=seed, max_iter=1000).fit(X_train, y_train)
    onehund_hl_score = clf.score(X_test, y_test)
    print("accuracy for 100 hidden layers: ", onehund_hl_score)

    # Plot Data
    data = {'1 Hidden Layer': onehl_score,
            '100 Hidden Layers': onehund_hl_score}

    model_metrics = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(4, 3))
    plt.bar(model_metrics, values, color='blue',
            width=0.4)

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

    clf = MLPClassifier(hidden_layer_sizes=(1, 100),
                        learning_rate='constant',
                        learning_rate_init=0.00001,
                        random_state=seed, max_iter=3000).fit(X_train, y_train)

    ln00001_score = clf.score(X_test, y_test)
    print("accuracy for 1 hidden layer: ", ln00001_score)

    clf = MLPClassifier(hidden_layer_sizes=(1, 100),
                        learning_rate='constant',
                        learning_rate_init=0.1,
                        random_state=seed, max_iter=3000).fit(X_train, y_train)

    ln01_score = clf.score(X_test, y_test)
    print("accuracy for 1 hidden layer: ", ln01_score)

    # Plot Data
    data = {'Learning Rate: 0.00001': ln00001_score,
            'Learning Rate: 0.1': ln01_score}

    model_metrics = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(4, 3))
    plt.bar(model_metrics, values, color='blue',
            width=0.4)

    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Voting Learning Rate Accuracy")
    plt.show()


""" c. Split the dataset into training, validation and testing subsets 
       by the ratio 60/20/20, try to find the best combinations for the 
       number of neurons and learning rate. Explain your method. """


def _1_1_c():
    # Get dataset
    dataset_directory = 'data'
    dataset = 'voting'
    schema, X, y = get_dataset(dataset, dataset_directory)

    data = np.column_stack((X, y))
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]

    # Split data into training and validation sets (60/20/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=seed)

    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (30,), (40,)],
        'learning_rate_init': [0.001, 0.01, 0.1, 1.0]
    }

    # Create MLPClassifier object
    clf = MLPClassifier(
        max_iter=1000, validation_fraction=0.2, random_state=seed)

    # Create GridSearchCV object and fit to training data
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best combination of hyperparameters and the corresponding score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)


""" d. Load the Iris dataset from sklearn.datasets, and split it
       into training and testing subsets by the ratio 70/30, 
       plot the confusion matrix and output the classification report. 
       (Tip: using the library) """


def _1_1_d():
    # Load the Iris dataset
    iris = load_iris()

    # Split the dataset into training and testing subsets using 70/30 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42)

    # Train an MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10),
                        max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = clf.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print the classification report
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def _2():
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'You could have done better.']

    # sentence labels
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # Define vocabulary
    vocab = set([word.lower() for doc in docs for word in doc.split()])

    # Create word to index dictionary
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    word_to_idx['<pad>'] = 0

    # Convert docs to tensor of word indices
    max_doc_length = max([len(doc.split()) for doc in docs])
    input_tensor = torch.tensor([[word_to_idx[word] for word in doc.lower().split(
    )] + [0]*(max_doc_length-len(doc.split())) for doc in docs], dtype=torch.int64)

    # Convert labels to tensor
    label_tensor = torch.tensor(labels, dtype=torch.int64)

    # Define RNN model
    class RNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            output, hidden = self.rnn(embedded)
            last_hidden = hidden[-1]
            return self.fc(last_hidden)

    # Initialize model
    vocab_size = len(word_to_idx)
    embedding_dim = 50
    hidden_dim = 100
    output_dim = 2
    model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train model
    n_epochs = 100
    batch_size = 5
    for epoch in range(n_epochs):
        for i in range(0, len(docs), batch_size):
            batch_input = input_tensor[i:i+batch_size]
            batch_labels = label_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Test model
    test_doc = 'Good effort'
    test_input = torch.tensor([[word_to_idx[word.lower()] for word in test_doc.split(
    )] + [0]*(max_doc_length-len(test_doc.split()))], dtype=torch.int64)
    output = model(test_input)
    prediction = torch.argmax(output).item()
    print(f'Test doc: {test_doc}, Prediction: {prediction}')


def _3():
    # Convolutional Neural Network (CNN)
    print("Working on this")


if __name__ == "__main__":
    # main()
    # print("1-1-a:")
    # _1_1_a()

    # print("1-1-b:")
    # _1_1_b()

    # print("1-1-c:")
    # _1_1_c()

    # print("1-1-d:")
    # _1_1_d()

    # print("2:")
    # _2()

    print("3:")
    _3()
