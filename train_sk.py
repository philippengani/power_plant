
from sklearn.ensemble import RandomForestRegressor
import sys
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


altPowerPlantDF = pd.read_csv("ccpp.csv")

# Split the data into training and test sets. (0.75, 0.25) split.
trainDF, testDF = train_test_split(altPowerPlantDF)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = trainDF.drop(["PE"], axis=1)
test_x = testDF.drop(["PE"], axis=1)
train_y = trainDF[["PE"]]
test_y = testDF[["PE"]]

mlflow.sklearn.autolog()


def mlflow_run(params, run_name="Tracking Experiment: TensorFlow - CNN "):
    with mlflow.start_run(run_name=" Power output prediction") as run:
        # Define the ML pipeline

        max_depth = params['max_depth']
        n_estimators = params['n_estimators']
        rf = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=0)
        rf.fit(train_x, train_y)

        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.sklearn.log_model(rf, "model", input_example=trainDF.head())

        # Evaluate predictions
        predicted_qualities = rf.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("RandomForest model (mex_depth=%f, n_estimators=%f):" % (max_depth, n_estimators))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Get the run and experiement id
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log artifact
        plt.clf()
        trainDF.hist(column="PE", bins=100)
        figPath = "normal.png"
        plt.savefig(figPath)
        mlflow.log_artifact(figPath)

        return experiment_id, run_id

    # Use the model


if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    trees = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    params = {'max_depth': depth, 'n_estimators': trees}

    (exp_id, run_id) = mlflow_run(params)

    print(f"Finished Experiment id={exp_id} and run id = {run_id}")
