from sklearn.model_selection import train_test_split

def train_valid_test_split(train_data, test_data):
    # Prepare train/valid input values
    X = train_data.drop(["PassengerId", "Survived"], axis=1)

    # Prepare train/valid output values
    y = train_data[["Survived"]]

    # Split training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)

    # Prepare test input value
    X_test = test_data.drop("PassengerId", axis=1)
    
    return X_train, y_train, X_valid, y_valid, X_test