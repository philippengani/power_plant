import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path, test_size=0.2, random_state=123):
    # Load dataset
    data = pd.read_csv('ccpp.csv')
    # Process dataset
    x = data.drop(['PE'], axis=1)
    y = data['PE']
    # Log dataset parameters
    mlflow.log_param("dataset_path", path)
    mlflow.log_param("dataset_shape", data.shape)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("one_hot_encoding", True)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset(
        'ccpp.csv'
    )
    print(x_train.head())
    print(y_train.head())
