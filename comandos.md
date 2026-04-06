## Paso 0: Inicializar gcloud
gcloud init

## Paso 1: Definir variables base
PROJECT_ID=TU_PROJECT_ID
REGION=TU_REGION_CLOUD_RUN
REPO_NAME=TU_REPO_ARTIFACT
IMAGE_NAME=TU_NOMBRE_IMAGEN
TAG=v1
SERVICE_NAME=TU_SERVICIO_CLOUD_RUN
SA_NAME=TU_SERVICE_ACCOUNT
SA_EMAIL=$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com

PIPELINE_LOCAL_PATH=models/pipeline.joblib
PIPELINE_GCS_PATH=gs://TU_BUCKET/models/pipeline.joblib
BUCKET_NAME=TU_BUCKET
DATASET_ID=TU_DATASET_ID
BQ_LOCATION=TU_BQ_LOCATION

XTEST_TABLE=$PROJECT_ID.$DATASET_ID.iris_data
FEATURES_TABLE=$PROJECT_ID.$DATASET_ID.selected_features
PREDICTIONS_TABLE=$PROJECT_ID.$DATASET_ID.predictions
MODEL_NAME=TU_MODEL_NAME

## Paso 2: Subir el pipeline a Cloud Storage
gcloud storage cp $PIPELINE_LOCAL_PATH $PIPELINE_GCS_PATH --project $PROJECT_ID

## Paso 3: Crear repositorio de Artifact Registry (si no existe)
gcloud artifacts repositories create $REPO_NAME \
  --repository-format=docker \
  --location=$REGION \
  --project=$PROJECT_ID

## Paso 4: Configurar autenticación de Docker para Artifact Registry
gcloud auth configure-docker $REGION-docker.pkg.dev

## Paso 5: Construir y publicar la imagen
gcloud builds submit \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG \
  --project $PROJECT_ID

## Paso 6: Crear cuenta de servicio para Cloud Run (si no existe)
gcloud iam service-accounts create $SA_NAME \
  --display-name="Cloud Run Actividad 4 API" \
  --project $PROJECT_ID

## Paso 7: Asignar permisos a la cuenta de servicio
# Permiso minimo para leer el pipeline en un bucket especifico
gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectViewer" \
  --project=$PROJECT_ID

# Permiso para ejecutar jobs de BigQuery (este rol se asigna a nivel proyecto)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/bigquery.jobUser"

# Permisos minimos por tabla (evita dar acceso de editor a todo el dataset)
bq query --use_legacy_sql=false "GRANT \`roles/bigquery.dataViewer\` ON TABLE \`$XTEST_TABLE\` TO 'serviceAccount:$SA_EMAIL'"

bq query --use_legacy_sql=false "GRANT \`roles/bigquery.dataViewer\` ON TABLE \`$FEATURES_TABLE\` TO 'serviceAccount:$SA_EMAIL'"

bq query --use_legacy_sql=false "GRANT \`roles/bigquery.dataEditor\` ON TABLE \`$PREDICTIONS_TABLE\` TO 'serviceAccount:$SA_EMAIL'"

## Paso 8: Desplegar en Cloud Run (con variables de entorno)
gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG \
  --region $REGION \
  --project $PROJECT_ID \
  --platform managed \
  --service-account $SA_EMAIL \
  --set-env-vars PROJECT_ID=$PROJECT_ID,LOCATION=$BQ_LOCATION,PIPELINE_PATH=$PIPELINE_GCS_PATH,XTEST_TABLE=$XTEST_TABLE,FEATURES_TABLE=$FEATURES_TABLE,PREDICTIONS_TABLE=$PREDICTIONS_TABLE,MODEL_NAME=$MODEL_NAME \
  --no-allow-unauthenticated

## Paso 9: Dar permiso de invocación a un usuario
gcloud run services add-iam-policy-binding $SERVICE_NAME \
  --region $REGION \
  --project $PROJECT_ID \
  --member="user:<TU_EMAIL>@gmail.com" \
  --role="roles/run.invoker"

## Paso 10: Invocar el endpoint /predict_and_store
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

curl -X POST $SERVICE_URL/predict_and_store \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"