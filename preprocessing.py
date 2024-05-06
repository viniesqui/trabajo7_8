from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# leer dataset 
'''
El dataset es de cuantas ventas se generan como funcion del gasto en el mercadeo de television
viene de https://www.kaggle.com/datasets/devzohaib/tvmarketingcsv/data
y lo escogi porque es simple con solo dos columnas y es pequeño entonces se entrena rapido
'''
df = pd.read_csv('tvmarketing.csv')

# quitar la columna de prediccion 
X = df.drop('Sales', axis=1)
y = df['Sales']

# Splitear los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar
model = LinearRegression()
model.fit(X_train, y_train)

# serializarlo
joblib.dump(model, 'trained_model.pkl')