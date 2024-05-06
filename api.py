from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.externals import joblib
import numpy as np

# Definir el modelo de entrada
class InputData(BaseModel):
    feature1: float
    feature2: float
    # Add more features here

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