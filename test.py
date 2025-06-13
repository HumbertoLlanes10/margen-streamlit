import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("🔮 Predicción de Margen Mensual con Prophet (Datos Internos)")

# 👉 Tus datos directamente aquí
meses = [
    "2024-02-01", "2024-03-01",
    "2024-04-01", "2024-05-01"
]
margenes = [204.0, 200.0, 228.0, 219.0]

# Crear DataFrame para Prophet
df = pd.DataFrame({
    'ds': pd.to_datetime(meses),
    'y': margenes
})

st.subheader("📊 Márgenes históricos")
st.dataframe(df.rename(columns={'ds': 'Mes', 'y': 'Margen'}))

# Crear y entrenar modelo
model = Prophet()
model.fit(df)

# Predecir próximos 3 meses
future = model.make_future_dataframe(periods=3, freq='MS')
forecast = model.predict(future)

# Extraer predicción
prediccion = forecast[['ds', 'yhat']].tail(3)
prediccion.rename(columns={'ds': 'Mes', 'yhat': 'Margen_Predicho'}, inplace=True)

st.subheader("📈 Predicción para los próximos 3 meses")
st.dataframe(prediccion)

# Mostrar gráfico
fig = model.plot(forecast)
st.pyplot(fig)