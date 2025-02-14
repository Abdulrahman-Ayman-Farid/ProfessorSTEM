from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the trained pipeline
with open("model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_level(
    age: int = Form(...),
    school_level: str = Form(...),
    science_grade: float = Form(...),
):
    # Prepare input as a DataFrame with the correct column names
    input_data = pd.DataFrame([{
        "Age": age,
        "School_Level": school_level,
        "Science_Grade": science_grade
    }])

    # Predict using the model pipeline
    prediction = model.predict(input_data)[0]
    return {"prediction": prediction}
