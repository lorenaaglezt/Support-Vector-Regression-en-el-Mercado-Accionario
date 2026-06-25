# Predicción de Precios de Acciones del Mercado Accionario

## Objetivo
El objetivo de este proyecto es predecir los precios de cierre de un conjunto de acciones del mercado (HPE, AVGO, CSCO, DELL, GOOGL, MU) empleando modelos de aprendizaje automático, específicamente **Support Vector Regression (SVR)** y **Ridge Regression**, evaluando distintas configuraciones y kernels (Lineal y RBF). Además de comparar el rendimiento estático, se implementa una estrategia de validación en el tiempo usando una ventana deslizante para simular un escenario de predicción en producción.

## Resumen

El sistema funciona siguiendo los siguientes pasos:
1. **Obtención de Datos:** Se descargan los datos históricos de los activos utilizando `yfinance` para el rango temporal de Enero 2020 a Mayo 2026 y se almacenan en una caché local (`data_cache/`).
2. **Indicadores Técnicos:** A partir del precio y volumen, se construyen cinco indicadores que alimentan el modelo: Media Móvil Simple (SMA), Media Móvil Ponderada (WMA), Índice de Fuerza Relativa (RSI), Oscilador de Acumulación/Distribución (ADO) y Rango Verdadero Promedio (ATR).
3. **Optimización y Validación:** Se evalúan los modelos (Ridge y SVR) utilizando un esquema de validación cruzada para series de tiempo (`TimeSeriesSplit`) para encontrar los mejores hiperparámetros sin espiar el futuro.
4. **Ventana Deslizante (Rolling Window):** Se simula el comportamiento continuo reentrenando el modelo periódicamente con los datos más recientes.
5. **Evaluación y Visualización:** Finalmente, se evalúan las métricas de rendimiento (RMSE y MAPE) y se generan automáticamente gráficas de calor y comparativas en la carpeta `images/`.

## Paqueterías Necesarias
Para ejecutar el proyecto de forma correcta, necesitas contar con Python 3 e instalar las siguientes dependencias:

- `yfinance`: Para la descarga de datos bursátiles.
- `numpy`: Para las operaciones matriciales y matemáticas.
- `pandas`: Para la manipulación de dataframes e indicadores.
- `matplotlib`: Para la generación de gráficas.
- `seaborn`: Para facilitar el diseño de mapas de calor (heatmaps).
- `scikit-learn`: Para el preprocesamiento (MinMaxScaler), los modelos de SVR y Ridge, y la validación cruzada.

Puedes instalar las dependencias con el siguiente comando:
```bash
pip install yfinance numpy pandas matplotlib seaborn scikit-learn
```

## Herramienta de Volatilidad

El proyecto incluye un script complementario (`volatilidad.py`) diseñado para justificar el desempeño de los modelos predictivos. Esta herramienta calcula:
- **Volatilidad Histórica Anualizada**
- **Beta** (riesgo frente al índice S&P 500)
- **Rango Diario Promedio**

Una volatilidad alta en ciertas acciones suele limitar la capacidad de los modelos de *Machine Learning* para minimizar el RMSE debido al ruido estocástico del mercado.

## Cómo ejecutar el programa

1. Clona o descarga el repositorio.
2. Abre una terminal y navega hasta el directorio raíz del proyecto.
3. Ejecuta el script principal para el entrenamiento y predicción:
   ```bash
   python proyecto_final.py
   ```
4. Durante la ejecución, el script principal:
   - Descargará los datos si no existen en `data_cache/`.
   - Calculará los indicadores y dividirá los datos.
   - Ejecutará la búsqueda de hiperparámetros (lo cual puede tomar un poco de tiempo).
   - Generará métricas en consola y guardará las figuras comparativas en la carpeta `images/`.
5. (Opcional) Ejecuta el análisis de volatilidad:
   ```bash
   python volatilidad.py
   ```