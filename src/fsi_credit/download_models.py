import json

import mlflow
import os
from mlflow.models import Model

mlflow.set_registry_uri("databricks-uc")

uc_catalog = "uc_sriharsha_jana"
uc_schema = "fsi_credit_data"
model_registry_name = "fsi_credit_defaulter"
model_alias = "champion"
model_version_uri = f"models:/{uc_catalog}.{uc_schema}.{model_registry_name}@{model_alias}"
downloaded_model_path = mlflow.artifacts.download_artifacts(model_version_uri, dst_path="./src/fsi_credit/models/")
print('downloaded models path -', downloaded_model_path)
print('files in downloaded path -', ','.join(os.listdir(downloaded_model_path)))

input_example = Model.load(downloaded_model_path).load_input_example(downloaded_model_path)
input_example = input_example.fillna(0)
dataset = input_example.to_dict(orient='index')
print(json.dumps(dataset))
