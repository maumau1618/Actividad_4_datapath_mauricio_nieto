from fastapi import FastAPI
from datetime import datetime, timezone
from google.auth import default as google_auth_default
from google.cloud import bigquery, storage
import joblib
import os
import pandas as pd
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen
import uuid

app = FastAPI()
ruta_actual = os.getcwd()

project = os.environ.get("PROJECT_ID")
location = os.environ.get("LOCATION")
pipeline_path = os.environ.get("PIPELINE_PATH")
source_x_test_table = os.environ.get("XTEST_TABLE")
features_table = os.environ.get("FEATURES_TABLE")
predictions_table = os.environ.get("PREDICTIONS_TABLE")
model_name = os.environ.get("MODEL_NAME")
service_name = os.environ.get("K_SERVICE")

METADATA_SERVICE_ACCOUNT_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
)


def get_gcp_credentials():
    # Usa ADC (Application Default Credentials): Cloud Run, local con gcloud auth o GOOGLE_APPLICATION_CREDENTIALS.
    credentials, _ = google_auth_default()
    return credentials


def get_creation_user() -> str:
    request = Request(
        METADATA_SERVICE_ACCOUNT_URL,
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urlopen(request, timeout=2) as response:
            return response.read().decode("utf-8").strip()
    except (URLError, TimeoutError):
        raise RuntimeError(
            "No fue posible obtener la identidad de servicio desde metadata server. "
            "Este servicio esta configurado solo para ejecutarse en Cloud Run."
        )


def get_created_by() -> str:
    if not service_name:
        raise RuntimeError(
            "La variable K_SERVICE no esta disponible. "
            "Este servicio esta configurado solo para ejecutarse en Cloud Run."
        )
    return service_name


def get_table_ref(client: bigquery.Client, table_name: str) -> bigquery.TableReference:
    # Soporta dataset.table y project.dataset.table.
    parts = table_name.split(".")
    if len(parts) == 2:
        dataset_id, table_id = parts
        return client.dataset(dataset_id).table(table_id)

    if len(parts) == 3:
        project_id, dataset_id, table_id = parts
        dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
        return dataset_ref.table(table_id)

    raise ValueError("La tabla debe tener formato dataset.tabla o project.dataset.tabla")

def get_selected_features():
    if not features_table:
        raise ValueError("Debe definir FEATURES_TABLE")

    credentials = get_gcp_credentials()
    client = bigquery.Client(project=project, location=location, credentials=credentials)
    query = f"SELECT features FROM `{features_table}`"
    features_df = client.query(query).to_dataframe()
    if "features" not in features_df.columns or features_df.empty:
        raise ValueError("FEATURES_TABLE no contiene datos en la columna 'features'")

    selected_features = []
    for value in features_df["features"].dropna().tolist():
        for feature in str(value).split(","):
            feature_name = feature.strip()
            if feature_name:
                selected_features.append(feature_name)

    selected_features = list(dict.fromkeys(selected_features))
    if not selected_features:
        raise ValueError("No se encontraron features validas en FEATURES_TABLE")

    return selected_features

def load_pipeline():
    # Cargamos el pipeline desde storage (gs://...) o ruta local.
    if not pipeline_path:
        raise ValueError("Debe definir PIPELINE_PATH")

    if pipeline_path.startswith("gs://"):
        credentials = get_gcp_credentials()
        path_without_scheme = pipeline_path.replace("gs://", "", 1)
        bucket_name, blob_name = path_without_scheme.split("/", 1)

        storage_client = storage.Client(project=project, credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            blob.download_to_filename(temp_path)
            pipeline = joblib.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return pipeline

    pipeline = joblib.load(pipeline_path)
    return pipeline

def load_data():
    # Cargamos el dataset de test desde BigQuery
    if not source_x_test_table:
        raise ValueError("Debe definir XTEST_TABLE")

    credentials = get_gcp_credentials()
    client = bigquery.Client(project=project, location=location, credentials=credentials)
    query = f"SELECT * FROM `{source_x_test_table}`"
    x_test = client.query(query).to_dataframe()
    return x_test

def prediccion_o_inferencia(pipeline, x_test_procesado):
    # Realizamos la predicción utilizando el modelo
    predicciones = pipeline.predict(x_test_procesado)
    return predicciones


def resolve_model_metadata(pipeline):
    # model_version debe venir del pipeline; model_name permite fallback por compatibilidad.
    resolved_model_name = getattr(pipeline, "model_name", None) or model_name
    resolved_model_version = getattr(pipeline, "model_version", None)

    if not resolved_model_name:
        raise RuntimeError(
            "No se encontro model_name en el pipeline ni en MODEL_NAME"
        )
    if not resolved_model_version:
        raise RuntimeError(
            "No se encontro model_version en el pipeline"
        )

    return resolved_model_name, resolved_model_version

def _feature_series(df: pd.DataFrame, preferred: str, alternatives: list[str]) -> pd.Series:
    for col_name in [preferred] + alternatives:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce")
    return pd.Series([None] * len(df), dtype="float64")


def _canonical_feature_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_expected_features(pipeline, selected_features: list[str]) -> list[str]:
    pipeline_features = getattr(pipeline, "feature_names", None)
    if pipeline_features:
        return [str(col) for col in pipeline_features]

    pipeline_features_in = getattr(pipeline, "feature_names_in_", None)
    if pipeline_features_in is not None and len(pipeline_features_in) > 0:
        return [str(col) for col in pipeline_features_in]

    if selected_features:
        return [str(col) for col in selected_features]

    raise RuntimeError("No se pudieron resolver las features esperadas por el modelo")


def build_model_input(pipeline, x_test: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    expected_features = _resolve_expected_features(pipeline, selected_features)

    actual_by_canonical = {}
    for col in x_test.columns:
        actual_by_canonical.setdefault(_canonical_feature_name(col), col)

    missing_features = []
    data = {}
    for feature_name in expected_features:
        resolved_col = actual_by_canonical.get(_canonical_feature_name(feature_name))
        if resolved_col is None:
            missing_features.append(feature_name)
            continue
        data[feature_name] = pd.to_numeric(x_test[resolved_col], errors="coerce")

    if missing_features:
        raise ValueError(f"Faltan columnas para inferencia: {missing_features}")

    return pd.DataFrame(data, index=x_test.index)


def store_predictions(
    predicciones,
    x_test,
    model_name_value,
    model_version_value,
    confidence_scores=None,
):
    # Almacenamos las predicciones en BigQuery
    if not predictions_table:
        raise ValueError("Debe definir PREDICTIONS_TABLE")

    credentials = get_gcp_credentials()
    client = bigquery.Client(project=project, location=location, credentials=credentials)
    table_ref = get_table_ref(client, predictions_table)

    predicciones_series = pd.Series(predicciones)
    predicted_class_id = pd.to_numeric(predicciones_series, errors="coerce").astype("Int64")
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_species = predicted_class_id.map(species_map)
    now_ts = datetime.now(timezone.utc)
    creation_user = get_creation_user()
    created_by = get_created_by()

    if confidence_scores is None:
        confidence_series = pd.Series([None] * len(predicciones_series), dtype="float64")
    else:
        confidence_series = pd.to_numeric(pd.Series(confidence_scores), errors="coerce")

    actual_species = x_test["species"].astype(str) if "species" in x_test.columns else None

    sepal_length_series = _feature_series(x_test, "sepal_length_cm", ["sepal length (cm)"])
    sepal_width_series = _feature_series(x_test, "sepal_width_cm", ["sepal width (cm)"])
    petal_length_series = _feature_series(x_test, "petal_length_cm", ["petal length (cm)"])
    petal_width_series = _feature_series(x_test, "petal_width_cm", ["petal width (cm)"])

    predictions_df = pd.DataFrame(
        {
            "prediction_id": [str(uuid.uuid4()) for _ in range(len(predicciones_series))],
            "request_id": str(uuid.uuid4()),
            "request_timestamp": now_ts,
            "model_name": model_name_value,
            "model_version": model_version_value,
            "sepal_length_cm": sepal_length_series,
            "sepal_width_cm": sepal_width_series,
            "petal_length_cm": petal_length_series,
            "petal_width_cm": petal_width_series,
            "predicted_species": predicted_species,
            "actual_species": actual_species,
            "predicted_class_id": predicted_class_id,
            "confidence_score": confidence_series,
            "prediction_status": "SUCCESS",
            "error_message": None,
            "creation_user": creation_user,
            "created_by": created_by,
            "load_date": now_ts,
            "updated_at": now_ts,
        }
    )

    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    # La tabla historica debe existir previamente; no permitir creacion implicita.
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_NEVER
    job_config.schema = [
        bigquery.SchemaField("prediction_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("request_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("request_timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("sepal_length_cm", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("sepal_width_cm", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("petal_length_cm", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("petal_width_cm", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("predicted_species", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("actual_species", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("predicted_class_id", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("confidence_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("prediction_status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("creation_user", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_by", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("load_date", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
    ]
    job = client.load_table_from_dataframe(predictions_df, table_ref, job_config=job_config)
    job.result()  # Esperamos a que el trabajo se complete


@app.get("/") 
def print_get():
    return {"mensaje": "Actividad 4 - Mauricio Nieto - MLOps 14"}

@app.post("/predict")
def predict():
    pipeline = load_pipeline()
    model_name_value, model_version_value = resolve_model_metadata(pipeline)
    x_test = load_data()
    selected_features = get_selected_features()

    x_test_model = build_model_input(pipeline, x_test, selected_features)
    predicciones = prediccion_o_inferencia(pipeline, x_test_model)

    confidence_scores = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(x_test_model)
        confidence_scores = probabilities.max(axis=1)

    store_predictions(
        predicciones,
        x_test,
        model_name_value,
        model_version_value,
        confidence_scores,
    )

    return {
        "mensaje": "Predicciones generadas y almacenadas en BigQuery",
        "total_predicciones": int(len(predicciones)),
        "tabla_destino": predictions_table,
    }
