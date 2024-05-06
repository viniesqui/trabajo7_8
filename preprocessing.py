from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd

# leer dataset 
df = pd.read_csv('your_dataset.csv')

# quitar la columna de prediccion 
X = df.drop('target_column', axis=1)
y = df['target_column']

# Splitear los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar
model = LinearRegression()
model.fit(X_train, y_train)

# serializarlo
joblib.dump(model, 'trained_model.pkl')