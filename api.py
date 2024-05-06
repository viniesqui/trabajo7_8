from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Definir el modelo de entrada
class InputData(BaseModel):
    Sales: float
    TV: float

app = FastAPI()

# agarrar el modelo entrenado en preprocessing
model = joblib.load('trained_model.pkl')

# Definir la ruta de la API y los CRUDS
@app.post('/predict')
def predict(data: InputData):
    input_data = np.array([data.feature1, data.feature2])  # mterle los features

    # predecir
    prediction = model.predict([input_data])

    return {'prediction': prediction[0]}