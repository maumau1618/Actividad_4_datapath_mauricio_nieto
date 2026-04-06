# Actividad 4: ML Inference Pipeline on Google Cloud Run

A FastAPI-based machine learning inference service that classifies Iris flowers using a pre-trained scikit-learn pipeline. The service runs on Google Cloud Run, integrates with BigQuery for data persistence, and uses Google Cloud Storage for model versioning.

## 📋 Overview

This project demonstrates a production-ready MLOps architecture with:
- **API Framework**: FastAPI with Uvicorn
- **Cloud Platform**: Google Cloud Run (serverless)
- **Data Storage**: Google BigQuery (structured queries)
- **Model Storage**: Google Cloud Storage (pipeline artifacts)
- **Authentication**: Application Default Credentials (ADC) via Cloud Run service account
- **Model Type**: Scikit-learn pipeline (LogisticRegression + StandardScaler)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Cloud Run Container                        │
│                  (FastAPI Application)                       │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐     │
│  │ Load Model   │→ │ Read Data    │→ │ Predict & Store│     │
│  │ from GCS     │  │ from BigQuery│  │ to BigQuery    │     │
│  └──────────────┘  └──────────────┘  └────────────────┘     │
│        ↓                  ↓                   ↓              │
└─────────┼──────────────────┼───────────────────┼─────────────┘
          │                  │                   │
     ┌────▼────┐     ┌───────▼──────┐    ┌──────▼───────┐
     │   GCS   │     │  BigQuery    │    │  BigQuery    │
     │ Bucket  │     │  Input Data  │    │ Predictions  │
     └─────────┘     └──────────────┘    └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Google Cloud Project with billing enabled
- `gcloud` CLI installed and configured
- Python 3.12+
- Docker (for local testing)

### Deployment Steps

1. **Set Environment Variables**
   ```bash
   export PROJECT_ID="your-gcp-project-id"
   export REGION="us-central1"
   export SERVICE_NAME="actividad-4-api"
   export BUCKET_NAME="your-gs-bucket"
   export DATASET_ID="your_dataset"
   export BQ_LOCATION="US"  # Multi-region location for your dataset
   ```

2. **Upload Model Pipeline**
   ```bash
   gcloud storage cp models/pipeline.joblib \
     gs://$BUCKET_NAME/models/pipeline.joblib \
     --project $PROJECT_ID
   ```

3. **Create Artifact Registry Repository**
   ```bash
   gcloud artifacts repositories create ml-pipeline-repo \
     --repository-format=docker \
     --location=$REGION \
     --project=$PROJECT_ID
   ```

4. **Configure Docker Authentication**
   ```bash
   gcloud auth configure-docker $REGION-docker.pkg.dev
   ```

5. **Build and Push Docker Image**
   ```bash
   gcloud builds submit \
     --tag $REGION-docker.pkg.dev/$PROJECT_ID/ml-pipeline-repo/iris-api:v1 \
     --project $PROJECT_ID
   ```

6. **Create Service Account**
   ```bash
   gcloud iam service-accounts create cloudrun-actividad-4-api \
     --display-name="Cloud Run ML Pipeline API" \
     --project $PROJECT_ID
   ```

7. **Grant Permissions**
   ```bash
   # Storage: Read-only access to pipeline artifact
   gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
     --member="serviceAccount:cloudrun-actividad-4-api@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.objectViewer" \
     --project=$PROJECT_ID

   # BigQuery: Permission to run queries
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:cloudrun-actividad-4-api@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/bigquery.jobUser"

   # BigQuery: Read access to input tables
   bq query --use_legacy_sql=false \
     "GRANT \`roles/bigquery.dataViewer\` ON TABLE \`$PROJECT_ID.$DATASET_ID.iris_data\` \
      TO 'serviceAccount:cloudrun-actividad-4-api@$PROJECT_ID.iam.gserviceaccount.com'"

   # BigQuery: Write access to predictions table
   bq query --use_legacy_sql=false \
     "GRANT \`roles/bigquery.dataEditor\` ON TABLE \`$PROJECT_ID.$DATASET_ID.predictions\` \
      TO 'serviceAccount:cloudrun-actividad-4-api@$PROJECT_ID.iam.gserviceaccount.com'"
   ```

8. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy $SERVICE_NAME \
     --image $REGION-docker.pkg.dev/$PROJECT_ID/ml-pipeline-repo/iris-api:v1 \
     --region $REGION \
     --project $PROJECT_ID \
     --platform managed \
     --service-account cloudrun-actividad-4-api@$PROJECT_ID.iam.gserviceaccount.com \
     --set-env-vars \
       PROJECT_ID=$PROJECT_ID,\
       LOCATION=$BQ_LOCATION,\
       PIPELINE_PATH=gs://$BUCKET_NAME/models/pipeline.joblib,\
       XTEST_TABLE=$PROJECT_ID.$DATASET_ID.iris_data,\
       FEATURES_TABLE=$PROJECT_ID.$DATASET_ID.selected_features,\
       PREDICTIONS_TABLE=$PROJECT_ID.$DATASET_ID.predictions,\
       MODEL_NAME=iris-logistic-regression \
     --no-allow-unauthenticated
   ```

9. **Grant Invocation Permission**
   ```bash
   gcloud run services add-iam-policy-binding $SERVICE_NAME \
     --region $REGION \
     --project $PROJECT_ID \
     --member="user:your-email@gmail.com" \
     --role="roles/run.invoker"
   ```

10. **Invoke the API**
    ```bash
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
      --region $REGION \
      --project $PROJECT_ID \
      --format='value(status.url)')
    
    TOKEN=$(gcloud auth print-identity-token)
    
    curl -X POST $SERVICE_URL/predict_and_store \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json"
    ```

## 📡 API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service_name": "actividad-4-api"
}
```

### `POST /predict_and_store`
Execute the full inference pipeline: load data, predict, and store results.

**Request:**
```bash
curl -X POST https://your-service-url/predict_and_store \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
  "mensaje": "Predicciones generadas y almacenadas en BigQuery",
  "total_predicciones": 150,
  "tabla_destino": "your-project.your_dataset.predictions"
}
```

## 🗄️ Data Model

### Input Tables

**`iris_data`** - Test samples for prediction
```sql
CREATE TABLE dataset.iris_data (
  sepal_length_cm FLOAT64,
  sepal_width_cm FLOAT64,
  petal_length_cm FLOAT64,
  petal_width_cm FLOAT64
);
```

**`selected_features`** - Metadata about expected features
```sql
CREATE TABLE dataset.selected_features (
  feature_name STRING
);
```

### Output Table

**`predictions`** - Inference results with audit trail
```sql
CREATE TABLE dataset.predictions (
  sample_id STRING,
  sepal_length_cm FLOAT64,
  sepal_width_cm FLOAT64,
  petal_length_cm FLOAT64,
  petal_width_cm FLOAT64,
  prediction INT64,
  probability_setosa FLOAT64,
  probability_versicolor FLOAT64,
  probability_virginica FLOAT64,
  model_name STRING,
  created_at TIMESTAMP,
  created_by STRING
);
```

## 🔐 Security Features

- **Least-Privilege IAM**: Service account has minimal permissions (read storage, read input tables, write output table only)
- **Application Default Credentials**: Uses Cloud Run's built-in identity (no hardcoded keys)
- **Metadata Server**: Service account email retrieved dynamically from GCP metadata service
- **Environment Variables**: All configuration externalized (no hardcoded values in code)
- **Feature Name Resolution**: Robust handling of schema variations between training and inference

## 🤖 Model Training

The model is trained using the `train_model.ipynb` notebook:

```python
# Pipeline components:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='saga', random_state=42))
])

# Trained on Iris dataset with features:
# - sepal_length, sepal_width, petal_length, petal_width
```

Model and scaler are saved to `models/` and synced to Cloud Storage.

## 📚 Dependencies

See `requirements.txt` for complete list. Key packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `google-cloud-bigquery` - BigQuery client
- `google-cloud-storage` - GCS client
- `pandas` - Data manipulation
- `scikit-learn` - ML models
- `joblib` - Model serialization
- `db-dtypes` - BigQuery type support for pandas

## 🔍 Troubleshooting

### Port Issues
Cloud Run injects a `PORT` environment variable. Ensure Uvicorn respects it:
```dockerfile
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

### BigQuery Location Mismatch
Set `LOCATION=US` for multi-region datasets. Region-specific datasets require matching location.

### Feature Name Mismatch
The API resolves feature name variations (e.g., "sepal_length_cm" vs "sepal length (cm)") using canonical comparison (alphanumeric-only, lowercase).

### Missing Permissions
Check service account roles with:
```bash
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:cloudrun-actividad-4-api@*"
```

##  Author

Mauricio Nieto

---

**Last Updated**: April 2026  
**Status**: Production-Ready ✅
