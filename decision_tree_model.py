import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(train_file, test_file):
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")
    return train_data, test_data

def select_features(data):
    features = ['Pclass_1', 'Sex', 'Age', 'Fare', 'Pclass_2', 'Pclass_3', 'Family_size', 'Title_1', 'Title_2', 'Title_3', 'Title_4', 'Emb_1', 'Emb_2', 'Emb_3']
    X = data[features]
    y = data['Survived']
    return X, y

def train_decision_tree(X_train, y_train):
    model = tree.DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def handle_missing_data(train_data, test_data):
    # Handle missing ages
    # Handle missing fare values
    # Handle missing embarked values
    # You can add your data handling code here
    return train_data, test_data

def predict_and_evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def make_predictions(model, test_data, features):
    test_features = test_data[features]
    test_predictions = model.predict(test_features)
    test_data['Survived'] = test_predictions
    return test_data

def plot_survived_vs_sex(train_data):
    sex_survived = pd.crosstab(train_data['Sex'].map({1: 'female', 0: 'male'}), train_data['Survived'])
    sex_survived.plot(kind='bar', stacked=True, color=['#CCCCCC', '#3F6D9B'])
    plt.title('Survived vs Sex')
    plt.xlabel('Sex')
    plt.ylabel('Number of Passengers')
    plt.legend(['Not Survived', 'Survived'])
    plt.show()

def main():
    train_data, test_data = load_data("C:\\dev\\summner 2023\\ML\\project\\train_data.csv", "C:\\dev\\summner 2023\\ML\\project\\test_data.csv")
    X_train, y_train = select_features(train_data)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = train_decision_tree(X_train_split, y_train_split)
    accuracy = predict_and_evaluate(model, X_val_split, y_val_split)
    print(f"Validation Accuracy: {accuracy:.2f}")
    train_data, test_data = handle_missing_data(train_data, test_data)
    test_data = make_predictions(model, test_data, X_train.columns)
    survived_passengers = test_data[test_data['Survived'] == 1]['PassengerId']
    print("Passengers who survived according to Decision Tree:")
    print(survived_passengers)
    plot_survived_vs_sex(train_data)

if __name__ == "__main__":
    main()
