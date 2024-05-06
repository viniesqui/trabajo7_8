from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

'''
Este es con esto se ejecuta el api. Usar el comando uvicorn main:app --reload
y usar el /docs para ver la documentacion de la api ya que es un POST y se necesita el dato de TV 
o hacerlo en postman o algo asi
'''


# Load the trained model
model = joblib.load('trained_model.pkl')

# Define the InputData model
class InputData(BaseModel):
    TV: float

# Create a FastAPI instance
app = FastAPI()

@app.post('/predict')
def predict(data: InputData):
    input_data = np.array([data.TV])  # Only use the features

    # Make prediction
    prediction = model.predict([input_data])

    return {'prediction': prediction[0]}