
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():

    names = ["Cultivar", "Alcohol", "Malic_acid", "Ash", "Alkalinity_ash", "Magnesium", "Phenols", "Flavanoids", "NF_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD", "Proline"]

    dataset = pd.read_csv('data_multi_class.csv', names = names)

    df = pd.DataFrame(dataset)
    
    scaler = MinMaxScaler()

    classLabelAttribute = ["Cultivar"]
    observationsLabelAttribute = ["Alcohol", "Malic_acid", "Ash", "Alkalinity_ash", "Magnesium", "Phenols", "Flavanoids", "NF_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD", "Proline"]

    df[observationsLabelAttribute] = scaler.fit_transform(df[observationsLabelAttribute].values.astype(float))

    target = df[classLabelAttribute]
    data = df[observationsLabelAttribute]

    train_size = 0.5
    test_size = 0.5
    random_state = 0

    train_x, test_x, train_y, test_y = train_test_split(data, target, train_size = train_size, test_size = test_size, random_state = random_state)
    
    n_estimators = 10

    rfc = RandomForestClassifier(n_estimators = n_estimators)

    trained_model = rfc.fit(train_x, train_y.values.ravel())

    predictions_train = trained_model.predict(train_x)
    predictions_test = trained_model.predict(test_x)

    print('Train Accuracy: {0}'.format(accuracy_score(train_y, predictions_train)))
    print('Test Accuracy: {0}'.format(accuracy_score(test_y, predictions_test)))

    print(pd.crosstab(index=test_y.values.ravel(), columns=predictions_test, rownames=["actual"], colnames=["predictions"]))

if __name__ == "__main__":
    main()
