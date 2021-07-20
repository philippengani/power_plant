
print("here")

import sys
import mlflow
import mlflow.spark
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler

# load the dataset
from pyspark.sql.types import *

# Custom Schema for Power Plant
customSchema = StructType([ \
    StructField("AT", DoubleType(), True), \
    StructField("V", DoubleType(), True), \
    StructField("AP", DoubleType(), True), \
    StructField("RH", DoubleType(), True), \
    StructField("PE", DoubleType(), True)])


'''altPowerPlantDF = sqlContext.read.format('com.databricks.spark.csv').options(delimiter='\t', header='true').load(
    "/databricks-datasets/power-plant/data", schema=customSchema)'''

altPowerPlantDF = spark.read.csv(
    "ccpp.csv",
    header=True,
    schema=customSchema
)

#altPowerPlantDF.show()
(trainDF, testDF) = altPowerPlantDF.randomSplit([.8, .2], seed=42)


def mlflow_run(params, run_name="Tracking Experiment: TensorFlow - CNN "):
    with mlflow.start_run(run_name=" Power output prediction") as run:
        # Define the ML pipeline
        vecAssembler = VectorAssembler(inputCols=["AT", "V", "AP", "RH"], outputCol="features")
        rf = RandomForestRegressor(labelCol="PE", maxBins=40, maxDepth=params['maxDepth'], numTrees=params['numTrees'])

        pipeline = Pipeline(stages=[vecAssembler, rf])
        pipelineModel = pipeline.fit(trainDF)

        # Log parameters
        mlflow.log_param("label", "PE")
        mlflow.log_param("max depth", params['maxDepth'])
        mlflow.log_param("num trees", params['numTrees'])

        # Log model
        mlflow.spark.log_model(pipelineModel, "model", input_example=trainDF.limit(5).toPandas())

        # Make predictions
        predDF = pipelineModel.transform(testDF)

        # Evaluate predictions
        regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="PE")
        rmse = regressionEvaluator.setMetricName("rmse").evaluate(predDF)
        r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)

        # Get the run and experiement id
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log artifact
        plt.clf()
        trainDF.toPandas().hist(column="PE", bins=100)
        figPath = "normal.png"
        plt.savefig(figPath)
        mlflow.log_artifact(figPath)

        return (experiment_id, run_id)

    # Use the model


'''if __name__ == "__main__":
    depth = float(sys.argv[1]) if len(sys.argv) > 1 else 3
    trees = float(sys.argv[2]) if len(sys.argv) > 2 else 10

    params = {'maxDepth': depth, 'numTrees': trees}

    (exp_id, run_id) = mlflow_run(params)

    print(f"Finished Experiment id={exp_id} and run id = {run_id}")'''
