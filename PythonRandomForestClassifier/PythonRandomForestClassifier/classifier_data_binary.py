import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def main():
    names = ["top-left-square", "top-middle-square", "top-right-square", "top-right-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "class"]

    dataset = pd.read_csv('data_binary.csv', names = names)

    df = pd.DataFrame(dataset)
    
    le = preprocessing.LabelEncoder()
    
    df = df.apply(le.fit_transform)

    classLabelAttribute = ["class"]
    observationsLabelAttribute = ["top-left-square", "top-middle-square", "top-right-square", "top-right-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square"]

    target = df[classLabelAttribute]
    data = df[observationsLabelAttribute]

    train_size = 0.5
    test_size = 0.5
    random_state = 0

    train_x, test_x, train_y, test_y = train_test_split(data, target, train_size = train_size, test_size = test_size, random_state = random_state)

    rfc = RandomForestClassifier(n_estimators = 10)

    trained_model = rfc.fit(train_x, train_y.values.ravel())

    predictions_train = trained_model.predict(train_x)
    predictions_test = trained_model.predict(test_x)

    print('Train Accuracy: {0}'.format(accuracy_score(train_y, predictions_train)))
    print('Test Accuracy: {0}'.format(accuracy_score(test_y, predictions_test)))

    print(pd.crosstab(index=test_y.values.ravel(), columns=predictions_test, rownames=["actual"], colnames=["predictions"]))

    visualize_classifier(rfc, train_x, train_y.values.ravel());

if __name__ == "__main__":
    main()
