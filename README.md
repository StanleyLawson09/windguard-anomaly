# WindGuard Anomaly Detection

An End-to-end machine learning Iot anomaly detection pipeline for predictive maintenance in an indistrual manufacturing context.

Project: From Model to Production


## Tech Stack
- Python 3.11
- sckikit-learn (Random Forest)
- pandas, numpy, joblib
- FastAPI + uvicorn


## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository, ID:601)


## Project Structure
- `data/` - dataset download code
- `model/` - training model
- `api/` - FastAPI prediction service
- `simulator/` - Iot stream simulator
- `monitoring/` - prediction log and dashboard


# Setup
pip install -r requirements.txt


## Run Order
python data/download_.py
python model/train_model.py
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
python simulator/simulate_stream.py
python monitoring/analyse_logs.py


## API Endpoints
- GET /health
- POST /predict
- POST /predict/batch
- GET /docs (Swagger UI)