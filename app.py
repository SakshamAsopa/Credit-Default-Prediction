from fastapi import FastAPI
import pandas as pd
import pickle
from nn_utils import predict_proba
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Credit Default Prediction API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


with open("nn_model.pkl", "rb") as f:
    params = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_default(data: dict):
    df = pd.DataFrame([data])

    # Ensure correct column order
    df = df.reindex(columns=columns, fill_value=0)

    X_scaled = scaler.transform(df)
    prob = float(predict_proba(X_scaled, params)[0][0])

    return {
        "default_probability": round(prob, 4),
        "default_prediction": int(prob > 0.5)
    }
