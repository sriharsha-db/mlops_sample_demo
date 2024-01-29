import json
import logging
import uuid
import mlflow
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np

SERVICE_NAME = 'fsi_credit_service'
# Configure logger
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["fsi_model"] = mlflow.pyfunc.load_model("./src/fsi_credit/models/")
    logger.info("loaded the model")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(title=SERVICE_NAME, docs_url="/", lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request):
    """Prediction endpoint.
    1. This should be a post request!
    2. Make sure to post the right data.
    """
    response_payload = None
    try:
        # Parse data
        input_dict = await request.json()
        input_df = pd.DataFrame.from_dict(input_dict, orient='index')
        input_df[input_df.select_dtypes(np.float64).columns] = input_df.select_dtypes(np.float64).astype(np.float32)
        input_df[input_df.select_dtypes(np.int64).columns] = input_df.select_dtypes(np.int64).astype(np.int32)

        print(input_df)
        print(input_df.dtypes)
        # Define UUID for the request
        request_id = uuid.uuid4().hex

        # Log input data
        logger.info(json.dumps({
            "service_name": SERVICE_NAME,
            "type": "InputData",
            "request_id": request_id,
            "data": input_df.to_json(orient='split')
        }))

        # Make predictions and log
        model_output = ml_models["fsi_model"].predict(input_df)
        print(type(model_output))
        output_dict = {"prediction": model_output.tolist()}
        # Log output data
        logger.info(json.dumps({
            "service_name": SERVICE_NAME,
            "type": "OutputData",
            "request_id": request_id,
            "data": output_dict
        }))

        # Make response payload
        response_payload = jsonable_encoder(output_dict)
    except Exception as e:
        logger.error("error in the fsi service ", e)
        response_payload = {
            "error": str(e)
        }

    return response_payload
