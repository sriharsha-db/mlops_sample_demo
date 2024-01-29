FROM python:3.10.13-slim-bullseye
ARG DATABRICKS_HOST_ARG
ARG DATABRICKS_TOKEN_ARG

COPY ../src src
COPY ../requirements.txt requirements-dev.txt

ENV DATABRICKS_HOST=$DATABRICKS_HOST_ARG
ENV DATABRICKS_TOKEN=$DATABRICKS_TOKEN_ARG

RUN pip install --no-cache-dir -r requirements.txt \
    && python src/fsi_credit/download_models.py \
    && pip install --no-cache-dir -r src/fsi_credit/models/requirements.txt

EXPOSE 8080
CMD ["uvicorn","src.fsi_credit.realtime_inference:app","--host","0.0.0.0","--port","8080"]