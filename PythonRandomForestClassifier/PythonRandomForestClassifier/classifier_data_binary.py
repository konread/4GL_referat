import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def main():
    names = ["top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "class"]

    dataset = pd.read_csv('data_binary.csv', names = names)

    df = pd.DataFrame(dataset)
    
    le = preprocessing.LabelEncoder()
    
    df = df.apply(le.fit_transform)

    classLabelAttribute = ["class"]
    observationsLabelAttribute = ["top-left-square", 
                                  "top-middle-square", 
                                  "top-right-square", 
                                  "top-right-square", 
                                  "middle-middle-square", 
                                  "middle-right-square", 
                                  "bottom-left-square", 
                                  "bottom-middle-square", 
                                  "bottom-right-square"]

    target = df[classLabelAttribute]
    data = df[observationsLabelAttribute]

    train_size = 0.75
    test_size = 0.25
    shuffle = True;

    train_x, test_x, train_y, test_y = train_test_split(data, 
                                                        target, 
                                                        train_size = train_size, 
                                                        test_size = test_size,
                                                        shuffle = shuffle)

    n_estimators = 500
    criterion = 'entropy'

    rfc = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion)

    trained_model = rfc.fit(train_x, train_y.values.ravel())

    #predictions_train = trained_model.predict(train_x)
    predictions_test = trained_model.predict(test_x)

    print("")
    
    print("Parametry dla funkcji train_test_split: ")
    print('train_size: {0} '.format(train_size), end='')
    print('test_size: {0} '.format(test_size), end='')
    print('shuffle: {0} '.format(shuffle), end='')
    
    print("")

    print("Parametry dla funkcji RandomForestClassifier:")
    print('n_estimators: {0} '.format(n_estimators), end='')
    print('criterion: {0} '.format(criterion), end='')

    print("")

    print(pd.crosstab(index=test_y.values.ravel(), columns=predictions_test, rownames=["Faktyczny"], colnames=["Przewidywania"]))

    #print('Wynik klasyfikacji dla zbioru uczącego: {0}'.format(accuracy_score(train_y, predictions_train)))
    print('Wynik klasyfikacji: {0}'.format(accuracy_score(test_y, predictions_test)))

if __name__ == "__main__":
    main()
