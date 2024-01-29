import mlflow
import pandas as pd
from argparse import ArgumentParser, Namespace
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

temp_local_path: str = "/tmp/local_train_data"

spark = SparkSession.builder \
    .appName("fsi_feature_processing") \
    .getOrCreate()


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description='parse the command line arguments for fsi model training')
    parser.add_argument('--mlflow_exp_name', type=str, required=True)
    parser.add_argument('--uc_catalog', type=str, required=True)
    parser.add_argument('--uc_schema', type=str, required=True)
    parser.add_argument('--feature_store', type=str, required=True)
    parser.add_argument('--model_registry_name', type=str, required=True)
    return parser.parse_args()


def __get_subsampled_data(catalog, schema):
    credit_bureau_label = (spark.table(f"{catalog}.{schema}.credit_bureau_gold_source")
                           .withColumn("defaulted", when(col("CREDIT_DAY_OVERDUE") > 60, 1)
                                       .otherwise(0))
                           .select(col("cust_id").cast("int"), "defaulted"))
    major_df = credit_bureau_label.filter(col("defaulted") == 0)
    minor_df = credit_bureau_label.filter(col("defaulted") == 1)
    oversampled_df = minor_df.union(minor_df)
    undersampled_df = major_df.sample(oversampled_df.count() / major_df.count() * 3, 42)
    train_df = undersampled_df.unionAll(oversampled_df).na.fill(0)
    train_df.write.format("delta").mode('overwrite').option("overwriteSchema", "true").save(temp_local_path)


def __read_data_feature_store(uc_table):
    source_df = spark.read.format("delta").load(temp_local_path)
    fe = FeatureEngineeringClient()
    feature_lookups = [FeatureLookup(table_name=uc_table,
                                     lookup_key="cust_id")]
    training_set = fe.create_training_set(df=source_df,
                                          feature_lookups=feature_lookups,
                                          label="defaulted",
                                          exclude_columns=['cust_id'])
    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop("defaulted", axis=1)
    y = training_pd["defaulted"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test


def __train_model(x_train, x_test, y_train, y_test, exp_id, params):
    mlflow.sklearn.autolog(disable=True)
    num_imputers = [("impute_mean", SimpleImputer(), list(x_train.columns))]
    numerical_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
        ("imputers", ColumnTransformer(num_imputers)),
        ("standardizer", StandardScaler()),
    ])

    numerical_transformers = [("numerical", numerical_pipeline, list(x_train.columns))]
    preprocessor = ColumnTransformer(numerical_transformers, remainder="passthrough", sparse_threshold=0)

    with mlflow.start_run(experiment_id=exp_id) as run:
        xgbc_classifier = XGBClassifier(**params)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", xgbc_classifier),
        ])
        mlflow.sklearn.autolog(log_input_examples=True, log_models=True, silent=True)

        model.fit(x_train, y_train)

        target_col = "defaulted"
        pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=x_test.assign(**{str(target_col): y_test}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={"log_model_explainability": False,
                              "metric_prefix": "test_", "pos_label": 1}
        )

    return run.info.run_id


def __save_model(run_id, model_name):
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)


def main(linput_args: Namespace):
    exp_obj = mlflow.set_experiment(linput_args.mlflow_exp_name)

    __get_subsampled_data(linput_args.uc_catalog, linput_args.uc_schema)

    fs_table = f"{linput_args.uc_catalog}.{linput_args.uc_schema}.{linput_args.feature_store}"
    x_train, x_test, y_train, y_test = __read_data_feature_store(fs_table)

    train_params = {
        "learning_rate": 0.2,
        "max_depth": 10,
        "n_estimators": 29,
        "n_jobs": 100,
        "subsample": 0.2,
        "verbosity": 0,
        "random_state": 123,
    }
    run_id = __train_model(x_train, x_test, y_train, y_test, exp_obj.experiment_id, train_params)

    model_name = f"{linput_args.uc_catalog}.{linput_args.uc_schema}.{linput_args.model_registry_name}"
    __save_model(run_id, model_name)


if __name__ == "__main__":
    input_args = parse_args()
    main(input_args)
