# Universidad Nacional Autónoma de México
# Facultad de Ciencias

# Reconocimiento de Patrones y Aprendizaje Automatizado
# Proyecto Final. SVR y Ridge en el Mercado Accionario.

# Nombre: Lorena González Téllez
# No. de cuenta: 321288952

# ---------------------------------------------------------------------------------------

# LIBRERÍAS

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Semilla global para reproducibilidad
SEED = 42
np.random.seed(SEED)

# Modelos
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

# Validación cruzada y búsqueda de hiperparámetros
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, make_scorer

print("PROYECTO. Predicción de Precios de Acciones del Mercado Accionario.\n")

# ---------------------------------------------------------------------------------------

# OBTENCIÓN DE DATOS HISTÓRICOS

# Cada fila representa un día de mercado y contiene:
#   Open = precio de apertura
#   High = precio máximo del día
#   Low = precio mínimo del día
#   Close = precio de cierre
#   Volume = volumen negociado

ACCIONES = ['HPE', 'AVGO', 'CSCO', 'DELL', 'GOOGL', 'MU']
INICIO = '2020-01-01'
FIN = '2026-05-01'

# Directorio donde se guardan los CSVs de cada acción.
CACHE_DIR = 'data_cache'
import os
os.makedirs(CACHE_DIR, exist_ok=True)

datos_raw = {}

print("Datos Históricos del Mercado Accionario.\n")

for ticker in ACCIONES:
    cache_path = os.path.join(CACHE_DIR, f'{ticker}.csv')

    if os.path.exists(cache_path):
        # Cargar desde caché para garantizar reproducibilidad
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"- {ticker}: {len(df)} días (cargado desde caché)")
    else:
        # Primera ejecución: descargar y guardar
        df = yf.download(ticker, start=INICIO, end=FIN, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)
        print(f"- {ticker}: {len(df)} días, desde {df.index[0].date()} hasta {df.index[-1].date()} (guardado en caché)")

    datos_raw[ticker] = df
    
# ---------------------------------------------------------------------------------------

# INDICADORES TÉCNICOS

P = 10  # Período para todos los indicadores

def calcular_indicadores(df):
    """
    Calcula los 5 indicadores técnicos sobre el DataFrame de precios.
    Recibe un DataFrame con columnas: Open, High, Low, Close, Volume.
    Devuelve un DataFrame con los indicadores como columnas.

    Indicadores calculados:
    - SMA  : Simple Moving Average (Media Móvil Simple)
    - WMA  : Weighted Moving Average (Media Móvil Ponderada)
    - RSI  : Relative Strength Index (Índice de Fuerza Relativa)
    - ADO  : Accumulation/Distribution Oscillator
    - ATR  : Average True Range (Rango Verdadero Promedio)
    """
    indicadores = pd.DataFrame(index=df.index)

    # SMA - promedio simple del precio de cierre de los últimos P días.
    indicadores['SMA'] = df['Close'].rolling(window=P).mean()

    # WMA - igual que SMA pero da más peso a los días más recientes.
    pesos = np.arange(1, P + 1)          # [1, 2, 3, ..., P]
    suma_pesos = pesos.sum()             # P*(P+1)/2
    indicadores['WMA'] = df['Close'].rolling(window=P).apply(
        lambda x: np.dot(x, pesos) / suma_pesos, raw=True
    )

    # RSI - mide si la acción está sobrecomprada o sobrevendida.
    delta     = df['Close'].diff()
    subidas   = delta.clip(lower=0)
    bajadas   = (-delta).clip(lower=0)

    avg_subidas = subidas.rolling(window=P).mean()
    avg_bajadas = bajadas.rolling(window=P).mean()

    rs  = avg_subidas / (avg_bajadas + 1e-10)
    indicadores['RSI'] = 100 - (100 / (1 + rs))

    # ADO - mide la fuerza de la tendencia.
    indicadores['ADO'] = (
        (df['High'] - df['Close'].shift(1)) /
        (df['High'] - df['Low'] + 1e-10)
    )

    # ATR -  mide la volatilidad del precio.
    high_low   = df['High'] - df['Low']
    high_prev  = (df['High'] - df['Close'].shift(1)).abs()
    low_prev   = (df['Low']  - df['Close'].shift(1)).abs()

    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    indicadores['ATR'] = tr.ewm(alpha=1/P, adjust=False).mean()

    # Variable objetivo: precio de cierre del día siguiente
    # Se desplaza -1 para que cada fila tenga el precio que queremos predecir.
    indicadores['Close_siguiente'] = df['Close'].shift(-1)
    indicadores['Close_actual'] = df['Close']

    # Eliminar filas con NaN (primeros P días no tienen indicadores completos
    # y el último día no tiene precio siguiente)
    indicadores.dropna(inplace=True)

    return indicadores

# Cálculo para los datos históricos

print("\nCálculo de Indicadores Técnicos.\n")

datos_procesados = {}
for ticker in ACCIONES:
    datos_procesados[ticker] = calcular_indicadores(datos_raw[ticker])
    print(f"- {ticker}: {len(datos_procesados[ticker])} muestras con indicadores")
    
# ---------------------------------------------------------------------------------------

# NORMALIZACIÓN DE DATOS

FEATURES = ['SMA', 'WMA', 'RSI', 'ADO', 'ATR'] # Variables predictoras X
TARGET = 'Close_siguiente' # Variable objetivo y

def preparar_datos(df_indicadores, proporcion_train=0.70):
    """
    Divide los datos en entrenamiento y prueba respetando
    el orden temporal (no se mezclan días), y normaliza las features.

    Retorna:
        X_train, X_test : matrices de indicadores normalizados
        y_train, y_test : vectores de precios objetivo
        y_test_real     : precios reales sin normalizar (para métricas)
        close_test      : precio de cierre actual en test (para random walk)
        scaler_X        : scaler ajustado (para invertir si se necesita)
    """
    n = len(df_indicadores)
    corte = int(n * proporcion_train)

    # División temporal: primero entrenamiento, después prueba
    train = df_indicadores.iloc[:corte]
    test  = df_indicadores.iloc[corte:]

    X_train_raw = train[FEATURES].values
    X_test_raw  = test[FEATURES].values
    y_train     = train[TARGET].values
    y_test      = test[TARGET].values

    # Normalizar features: scaler ajustado solo con train
    scaler_X = MinMaxScaler()
    X_train  = scaler_X.fit_transform(X_train_raw)
    X_test   = scaler_X.transform(X_test_raw)      # Aplica misma escala

    # Precio actual en test (para modelo base random walk)
    close_test = test['Close_actual'].values

    return X_train, X_test, y_train, y_test, close_test, scaler_X

# ---------------------------------------------------------------------------------------

# MÉTRICAS DE EVALUACIÓN

def calcular_rmse(y_real, y_pred):
    """
    RMSE - Root Mean Squared Error (Raíz del Error Cuadrático Medio)
    Mide el error promedio en las mismas unidades que el precio (dólares).
    """
    return np.sqrt(mean_squared_error(y_real, y_pred))


def calcular_mape(y_real, y_pred):
    """
    MAPE - Mean Absolute Percentage Error (Error Porcentual Absoluto Medio)
    Mide el error como porcentaje del precio real.
    """
    return np.mean(np.abs((y_real - y_pred) / (y_real + 1e-10)))

# ---------------------------------------------------------------------------------------

# VALIDACIÓN CRUZADA

# TimeSeriesSplit respeta el orden temporal.
# Cada fold extiende el entrenamiento hacia atrás y prueba en datos
# más recientes, simulando cómo se usaría el modelo en producción.

N_SPLITS = 5    # Número de divisiones temporales
scorer_rmse = make_scorer(
    lambda y, yp: -calcular_rmse(y, yp)   # negativo porque GridSearch maximiza
)

# VALORES A EXPLORAR EN CADA MODELO

# Cada combinación será evaluada con validación cruzada temporal.
#
# Para SVR: C (penalización por errores fuera del tubo)
#           epsilon (tamaño del tubo de tolerancia)
#           gamma (alcance de influencia en cada punto para RBF) 
#
# Para Ridge: alpha (fuerza de regularización)

grilla_svr_lineal = {
    'C' : [0.01, 0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.05, 0.1, 0.5, 1.0]
} # 125 entrenamientos

grilla_svr_rbf = {
    'C' : [0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma'  : [0.001, 0.01, 0.1, 1.0]
} # 240 entrenamientos

grilla_ridge_lineal = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
} # 35 entrenamientos

grilla_ridge_rbf = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'gamma': [0.001, 0.01, 0.1, 1.0]
} # 120 entrenamientos

# FUNCIÓN DE BÚSQUEDA CON VALIDACIÓN CRUZADA

def buscar_hiperparametros(nombre, modelo_clase, grilla, X_train, y_train,
                            n_splits=N_SPLITS):
    """
    Ejecuta GridSearchCV con TimeSeriesSplit para encontrar los mejores
    hiperparámetros de un modelo dado.

    Parámetros:
        nombre : nombre del modelo
        modelo_clase: instancia del modelo (SVR, Ridge, KernelRidge)
        grilla : diccionario con los valores a explorar
        X_train : features de entrenamiento normalizadas
        y_train : precios objetivo de entrenamiento

    Retorna:
        mejor_modelo : modelo reentrenado con los mejores parámetros
        mejores_params : diccionario con los parámetros óptimos
        mejor_rmse : RMSE en validación cruzada con esos parámetros
        resultados_cv : DataFrame con todos los resultados de la búsqueda
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    busqueda = GridSearchCV(
        estimator  = modelo_clase,
        param_grid = grilla,
        cv         = tscv,
        scoring    = scorer_rmse,
        n_jobs     = 1,          # n_jobs=1 garantiza orden determinista
        refit      = True,       # Reentrena con los mejores params en todo X_train
        verbose    = 0
    )
    busqueda.fit(X_train, y_train)

    mejor_rmse    = -busqueda.best_score_   # Quitar el negativo
    mejores_params = busqueda.best_params_
    mejor_modelo   = busqueda.best_estimator_

    # Tabla con todos los resultados ordenados por RMSE
    resultados_cv = pd.DataFrame(busqueda.cv_results_)
    resultados_cv['rmse_cv'] = -resultados_cv['mean_test_score']
    resultados_cv = resultados_cv.sort_values('rmse_cv')

    print(f"Mejores parámetros en {nombre} = {mejores_params}")

    return mejor_modelo, mejores_params, mejor_rmse, resultados_cv

# ---------------------------------------------------------------------------------------

# DEFINICIÓN DEL EXPERIMENTO CON LOS MODELOS A COMPARAR

def experimento_modelos(ticker, df_ind):
    """
    Ejecuta el experimento completo para una acción.
    
    1. Prepara y divide los datos.
    2. Evalúa Random Walk.
    3. Para cada modelo (Ridge Lineal, Ridge RBF, SVR Lineal, SVR RBF):
       a) Entrena y evalúa con parámetros por defecto.
       b) Entrena y evalúa con búsqueda de hiperparámetros.
    4. Compara métricas (RMSE, MAPE) y complejidad (# parámetros).
    """
    
    print(f"\nACCIÓN: {ticker}\n")

    # Preparar datos
    X_train, X_test, y_train, y_test, close_test, _ = preparar_datos(df_ind)
    n_features = X_train.shape[1]

    print(f"- Entrenamiento: {len(X_train)} días")
    print(f"- Prueba: {len(X_test)} días\n")

    resultados = {}

    # RANDOM WALK
    pred_rw = close_test
    resultados['Random Walk'] = {
        'RMSE'    : calcular_rmse(y_test, pred_rw),
        'MAPE'    : calcular_mape(y_test, pred_rw),
        'n_params': 0,
        'preds'   : pred_rw
    }

    # RIDGE LINEAL SIN OPTIMIZACIÓN alpha=1.0
    m_rl_def = Ridge(alpha=1.0)
    m_rl_def.fit(X_train, y_train)
    pred_rl_def = m_rl_def.predict(X_test)
    n_params_rl = 1 + n_features + 1  # alpha + coefs + intercepto
    
    resultados['Ridge Lineal'] = {
        'RMSE'    : calcular_rmse(y_test, pred_rl_def),
        'MAPE'    : calcular_mape(y_test, pred_rl_def),
        'n_params': n_params_rl,
        'preds'   : pred_rl_def
    }

    # RIDGE LINEAL CON OPTIMIZACIÓN
    m_rl_opt, p_rl, _, _ = buscar_hiperparametros(
        'Ridge Lineal', Ridge(), grilla_ridge_lineal, X_train, y_train
    )
    pred_rl_opt = m_rl_opt.predict(X_test)
    
    resultados['Ridge Lineal con VC'] = {
        'RMSE'    : calcular_rmse(y_test, pred_rl_opt),
        'MAPE'    : calcular_mape(y_test, pred_rl_opt),
        'n_params': n_params_rl,
        'preds'   : pred_rl_opt
    }

    # RIDGE CON KERNEL RBF SIN OPTIMIZACIÓN alpha=1.0 y gamma=0.1
    m_rk_def = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
    m_rk_def.fit(X_train, y_train)
    pred_rk_def = m_rk_def.predict(X_test)
    n_params_rk = 2 + len(X_train)  # alpha, gamma + N coeficientes duales
    
    resultados['Ridge con Kernel RBF'] = {
        'RMSE'    : calcular_rmse(y_test, pred_rk_def),
        'MAPE'    : calcular_mape(y_test, pred_rk_def),
        'n_params': n_params_rk,
        'preds'   : pred_rk_def
    }

    # RIDGE CON KERNEL RBF CON OPTIMIZACIÓN
    m_rk_opt, p_rk, _, _ = buscar_hiperparametros(
        'Ridge con Kernel RBF', KernelRidge(kernel='rbf'), grilla_ridge_rbf, X_train, y_train
    )
    pred_rk_opt = m_rk_opt.predict(X_test)
    
    resultados['Ridge con Kernel RBF y VC'] = {
        'RMSE'    : calcular_rmse(y_test, pred_rk_opt),
        'MAPE'    : calcular_mape(y_test, pred_rk_opt),
        'n_params': n_params_rk,
        'preds'   : pred_rk_opt
    }

    # SVR LINEAL SIN OPTIMIZACIÓN C=1.0 y epsilon=0.1
    m_sl_def = SVR(kernel='linear', C=1.0, epsilon=0.1)
    m_sl_def.fit(X_train, y_train)
    pred_sl_def = m_sl_def.predict(X_test)
    n_params_sl_def = 2 + len(m_sl_def.support_vectors_) # C + epsilon + vectores
    
    resultados['SVR Lineal'] = {
        'RMSE'    : calcular_rmse(y_test, pred_sl_def),
        'MAPE'    : calcular_mape(y_test, pred_sl_def),
        'n_params': n_params_sl_def,
        'preds'   : pred_sl_def
    }

    # SVR LINEAL CON OPTIMIZACIÓN
    m_sl_opt, p_sl, _, _ = buscar_hiperparametros(
        'SVR Lineal', SVR(kernel='linear'), grilla_svr_lineal, X_train, y_train
    )
    pred_sl_opt = m_sl_opt.predict(X_test)
    n_params_sl_opt = 2 + len(m_sl_opt.support_vectors_)
    
    resultados['SVR Lineal con VC'] = {
        'RMSE'    : calcular_rmse(y_test, pred_sl_opt),
        'MAPE'    : calcular_mape(y_test, pred_sl_opt),
        'n_params': n_params_sl_opt,
        'preds'   : pred_sl_opt
    }

    # SVR CON KERNEL RBF SIN OPTIMIZACIÓN C=1.0, epsilon=0.1 gamma=0.1
    m_sk_def = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma=0.1)
    m_sk_def.fit(X_train, y_train)
    pred_sk_def = m_sk_def.predict(X_test)
    n_params_sk_def = 3 + len(m_sk_def.support_vectors_) # C, epsilon, gamma + vectores
    
    resultados['SVR con Kernel RBF'] = {
        'RMSE'    : calcular_rmse(y_test, pred_sk_def),
        'MAPE'    : calcular_mape(y_test, pred_sk_def),
        'n_params': n_params_sk_def,
        'preds'   : pred_sk_def
    }

    # SVR CON KERNEL RBF CON OPTIMIZACIÓN
    m_sk_opt, p_sk, _, _ = buscar_hiperparametros(
        'SVR con Kernel RBF', SVR(kernel='rbf'), grilla_svr_rbf, X_train, y_train
    )
    pred_sk_opt = m_sk_opt.predict(X_test)
    n_params_sk_opt = 3 + len(m_sk_opt.support_vectors_)
    
    resultados['SVR con Kernel RBF y VC'] = {
        'RMSE'    : calcular_rmse(y_test, pred_sk_opt),
        'MAPE'    : calcular_mape(y_test, pred_sk_opt),
        'n_params': n_params_sk_opt,
        'preds'   : pred_sk_opt
    }

    # TABLA DE RESULTADOS
    print(f"\n {'Modelo':<30} | {'RMSE':>8} | {'MAPE':>7} | {'# Params':>8}")
    print(f" {'-'*66}")
    
    # Imprimimos en el orden en que fueron insertados en el diccionario
    # Añadimos un separador visual entre familias de modelos para mayor claridad
    for nombre, res in resultados.items():
        if nombre in ['Ridge con Kernel RBF', 'SVR Lineal', 'SVR con Kernel RBF']:
            print(f" {'-'*66}")

        print(f" {nombre:<30} | {res['RMSE']:>8.4f} | {res['MAPE']:>7.2%} | {res['n_params']:>8}")

    # Guardar parámetros óptimos y modelos para la ventana deslizante
    params_opt = {
        'Ridge Lineal con VC'      : p_rl,
        'Ridge con Kernel RBF y VC': p_rk,
        'SVR Lineal con VC'        : p_sl,
        'SVR con Kernel RBF y VC'  : p_sk,
    }

    return resultados, y_test, X_train, params_opt, df_ind
    
# ---------------------------------------------------------------------------------------

# ESTRATEGIA DE VENTANA DESLIZANTE (MOVING TRAINING WINDOW)
#
# Para cada día del conjunto de prueba, el modelo
# se reentrena usando únicamente los W días más recientes disponibles.
#
# Los hiperparámetros se fijan a los valores óptimos encontrados por VC
# en el experimento estático, lo que evita repetir la búsqueda en cada paso.
#
# Aquí se usa W=60 para garantizar suficiente muestra con 5 variables predictoras.

VENTANA_DIAS = 60   # Tamaño de la ventana de entrenamiento (días)

def prediccion_ventana_deslizante(df_ind, modelo_clase, params, ventana=VENTANA_DIAS,
                                   proporcion_train=0.70):
    """
    Estrategia de ventana deslizante: para cada punto de prueba t,
    reentrena el modelo con los días anteriores a t y
    predice el precio de cierre del día t.

    Los hiperparámetros (params) se fijan a los valores óptimos
    obtenidos previamente por validación cruzada en el modelo estático.

    Parámetros:
        df_ind   : DataFrame con indicadores y variable objetivo
        modelo_clase : instancia del modelo (SVR, Ridge, KernelRidge)
        params   : dict con los hiperparámetros óptimos
        ventana  : número de días en cada ventana de entrenamiento
        proporcion_train : fracción de datos usada en el experimento estático

    Retorna:
        preds    : array de predicciones para el conjunto de prueba
        y_test   : array de precios reales del conjunto de prueba
        n_test   : tamaño del conjunto de prueba
    """
    n_total = len(df_ind)
    n_train = int(n_total * proporcion_train)

    X_all = df_ind[FEATURES].values
    y_all = df_ind[TARGET].values

    # El scaler se ajusta una sola vez con el conjunto de entrenamiento
    # original para evitar filtración de información futura.
    scaler = MinMaxScaler()
    scaler.fit(X_all[:n_train])

    X_all_scaled = scaler.transform(X_all)

    preds = []

    for t in range(n_train, n_total):
        # Ventana: los `ventana` días inmediatamente anteriores a t
        inicio = max(0, t - ventana)
        X_win = X_all_scaled[inicio:t]
        y_win = y_all[inicio:t]

        # Instancia nueva con los parámetros óptimos y reentrenamiento
        modelo = modelo_clase.set_params(**params)
        modelo.fit(X_win, y_win)

        preds.append(modelo.predict(X_all_scaled[t:t+1])[0])

    return np.array(preds), y_all[n_train:], n_total - n_train


def experimento_ventana_deslizante(ticker, df_ind, params_opt, resultados_estaticos):
    """
    Ejecuta la estrategia de ventana deslizante para los cuatro modelos
    con hiperparámetros óptimos y compara con los modelos estáticos.

    Parámetros:
        ticker            : símbolo de la acción
        df_ind            : DataFrame con indicadores completos
        params_opt        : dict con parámetros óptimos de cada modelo con VC
        resultados_estaticos : dict con resultados del experimento estático

    Retorna:
        resultados_vd : dict con RMSE, MAPE y predicciones de ventana deslizante
    """
    print(f"\n- ACCIÓN: {ticker}  (W = {VENTANA_DIAS} días)\n")

    # Configuración de modelos y sus clases base
    modelos_vd = {
        'Ridge Lineal VD'      : (Ridge(),                    'Ridge Lineal con VC'),
        'Ridge RBF VD'         : (KernelRidge(kernel='rbf'),  'Ridge con Kernel RBF y VC'),
        'SVR Lineal VD'        : (SVR(kernel='linear'),       'SVR Lineal con VC'),
        'SVR RBF VD'           : (SVR(kernel='rbf'),          'SVR con Kernel RBF y VC'),
    }

    resultados_vd = {}

    for nombre_vd, (modelo_base, nombre_opt) in modelos_vd.items():
        params = params_opt[nombre_opt]
        preds, y_test, n_test = prediccion_ventana_deslizante(
            df_ind, modelo_base, params
        )
        rmse = calcular_rmse(y_test, preds)
        mape = calcular_mape(y_test, preds)

        # Comparación con el modelo estático equivalente
        rmse_est = resultados_estaticos[nombre_opt]['RMSE']
        mejora   = (rmse_est - rmse) / rmse_est * 100

        print(f"  {nombre_vd:<22} RMSE={rmse:.4f}  MAPE={mape:.2%}  "
              f"(vs estático: {mejora:+.1f}%)")

        resultados_vd[nombre_vd] = {
            'RMSE'        : rmse,
            'MAPE'        : mape,
            'preds'       : preds,
            'y_test'      : y_test,
            'modelo_orig' : nombre_opt,
        }

    return resultados_vd

# ---------------------------------------------------------------------------------------

# EJECUCIÓN COMPLETA CON ENTRENAMIENTO FIJO CON LAS ACCIONES SELECCIONADAS

print("\nEJECUCIÓN DE MODELOS CON ENTRENAMIENTO FIJO, CON Y SIN OPTIMIZACIÓN DE PARÁMETROS")
print("\nA continuación se aplican los modelos SVR y Ridge con kernels lineal y RBF.")

todos_resultados = {}
todos_y_test = {}
todos_X_train = {}
todos_params_opt = {}
todos_df_ind = {}

for ticker in ACCIONES:
    res, y_test, X_train, params_opt, df_ind = experimento_modelos(
        ticker, datos_procesados[ticker]
    )
    todos_resultados[ticker] = res
    todos_y_test[ticker] = y_test
    todos_X_train[ticker] = X_train
    todos_params_opt[ticker] = params_opt
    todos_df_ind[ticker] = df_ind

# PRESENTACIÓN DE RESULTADOS DE LOS MODELOS, CON Y SIN VALIDACIÓN CRUZADA Y ENTRENAMIENTO FIJO

print("\nRESULTADOS POR MODELO Y ACCIÓN")

modelos = list(todos_resultados[ACCIONES[0]].keys())
filas = []

for ticker in ACCIONES:
    for modelo in modelos:
        filas.append({
            'Acción': ticker, 'Modelo': modelo,
            'RMSE': todos_resultados[ticker][modelo]['RMSE'],
            'MAPE': todos_resultados[ticker][modelo]['MAPE'],
            '# Params': todos_resultados[ticker][modelo]['n_params']
        })

df_res = pd.DataFrame(filas)

# Pivots
pivot_rmse = df_res.pivot(index='Modelo', columns='Acción', values='RMSE')
pivot_rmse['PROMEDIO'] = pivot_rmse.mean(axis=1)
pivot_rmse = pivot_rmse.sort_values('PROMEDIO')

pivot_mape = df_res.pivot(index='Modelo', columns='Acción', values='MAPE')
pivot_mape['PROMEDIO'] = pivot_mape.mean(axis=1)
pivot_mape = pivot_mape.sort_values('PROMEDIO')

pivot_params = df_res.pivot(index='Modelo', columns='Acción', values='# Params')
pivot_params['PROMEDIO'] = pivot_params.mean(axis=1)

# Imprimimos resultados

print("\nRMSE POR ACCIÓN Y MODELO")
print(pivot_rmse.round(4).to_string())

print("\nMAPE POR ACCIÓN Y MODELO")
print((pivot_mape * 100).round(2).to_string())

print("\nCOMPLEJIDAD DE MODELOS")
print(pivot_params[['PROMEDIO']].round(0).astype(int).to_string())

# ---------------------------------------------------------------------------------------

# EJECUCIÓN DE LOS EXPERIMENTOS CON ENTRENAMIENTO DE VENTANA DESLIZANTE
# Se usan los hiperparámetros obtenidos anteriormente

print("\nEJECUCIÓN DE MODELOS CON VENTANA DESLIZANTE\n")
print("El entrenamiento usa los parámetros optimizados antes con validación cruzada.")

todos_resultados_vd = {}

for ticker in ACCIONES:
    todos_resultados_vd[ticker] = experimento_ventana_deslizante(
        ticker,
        todos_df_ind[ticker],
        todos_params_opt[ticker],
        todos_resultados[ticker]
    )

NOMBRES_VD = ['Ridge Lineal VD', 'Ridge RBF VD', 'SVR Lineal VD', 'SVR RBF VD']
NOMBRES_EST_VC = ['Ridge Lineal con VC', 'Ridge con Kernel RBF y VC',
                  'SVR Lineal con VC', 'SVR con Kernel RBF y VC']

filas_vd = []
for ticker in ACCIONES:
    for nombre_vd in NOMBRES_VD:
        res = todos_resultados_vd[ticker][nombre_vd]
        filas_vd.append({
            'Acción': ticker,
            'Modelo': nombre_vd,
            'RMSE'  : res['RMSE'],
            'MAPE'  : res['MAPE'],
        })

df_vd = pd.DataFrame(filas_vd)
pivot_rmse_vd = df_vd.pivot(index='Modelo', columns='Acción', values='RMSE')
pivot_rmse_vd['PROMEDIO'] = pivot_rmse_vd.mean(axis=1)

pivot_mape_vd = df_vd.pivot(index='Modelo', columns='Acción', values='MAPE')
pivot_mape_vd['PROMEDIO'] = pivot_mape_vd.mean(axis=1)

# PRESENTACIÓN DE RESULTADOS EN LA APLICACIÓN DE VENTANA DESLIZANTE

print("\nRMSE POR ACCIÓN Y MODELO CON VENTANA DESLIZANTE")
print(pivot_rmse_vd.round(4).to_string())

print("\nMAPE POR ACCIÓN Y MODELO CON VENTANA DESLIZANTE")
print((pivot_mape_vd * 100).round(2).to_string())


# ---------------------------------------------------------------------------------------

# VISUALIZACIÓN DE RESULTADOS

# Paleta de colores fija para cada modelo
COLORES_MODELOS = {
    'Random Walk'              : 'dimgray',
    'Ridge Lineal'             : "#1DC9F8",
    'Ridge Lineal con VC'      : "#05429f",
    'Ridge con Kernel RBF'     : "#18f527",
    'Ridge con Kernel RBF y VC': "#FD1CBD",
    'SVR Lineal'               : "#FBEF11",
    'SVR Lineal con VC'        : "#F0840F",
    'SVR con Kernel RBF'       : "#AB0BF0",
    'SVR con Kernel RBF y VC'  : "#EB0505",
}

# Estilos de línea: sólido para sin VC, discontinuo para con VC
ESTILOS_LINEA = {
    'Random Walk'              : ':',
    'Ridge Lineal'             : '-',
    'Ridge Lineal con VC'      : '-',
    'Ridge con Kernel RBF'     : '-',
    'Ridge con Kernel RBF y VC': '-',
    'SVR Lineal'               : '-',
    'SVR Lineal con VC'        : '-',
    'SVR con Kernel RBF'       : '-',
    'SVR con Kernel RBF y VC'  : '-',
}

print("\nGenerando visualizaciones de resultados.")


# GRÁFICA 1: RANKING DE MODELOS POR RMSE PROMEDIO

# Barras horizontales ordenadas de menor a mayor RMSE promedio.
# Permite identificar de un vistazo qué modelo predice mejor en promedio.

def graficar_ranking_rmse(pivot_rmse):
    """
    Gráfica de barras horizontales con el RMSE promedio de cada modelo,
    ordenado de mejor (menor) a peor (mayor).
    """
    rmse_ord = pivot_rmse['PROMEDIO'].sort_values(ascending=True)
    colores  = [COLORES_MODELOS[m] for m in rmse_ord.index]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(rmse_ord.index, rmse_ord.values,
                   color=colores, edgecolor='black', linewidth=0.6, height=0.6)

    # Etiqueta con el valor exacto al final de cada barra
    for bar, val in zip(bars, rmse_ord.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_xlabel('RMSE Promedio (dólares)', fontsize=11)
    ax.set_title('Ranking de Modelos por RMSE Promedio',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, rmse_ord.max() * 1.15)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()   # El mejor modelo queda arriba
    plt.tight_layout()
    plt.savefig('images/grafica_1_ranking_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("- grafica_1_ranking_rmse.png")


graficar_ranking_rmse(pivot_rmse)


# GRÁFICA 2: HEATMAPS DE RMSE Y MAPE POR MODELO Y ACCIÓN

# Muestra simultáneamente el error de cada modelo en cada acción.
# Permite detectar si un modelo falla sistemáticamente en alguna acción
# y si hay consistencia entre RMSE y MAPE.

def graficar_heatmaps(pivot_rmse, pivot_mape):
    """
    Dos heatmaps lado a lado: RMSE (izquierda) y MAPE en % (derecha).
    Las filas son los modelos y las columnas las acciones.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Error por Modelo y Acción', fontsize=13, fontweight='bold')

    # Heatmap RMSE (sin columna PROMEDIO para que no distorsione la escala de color)
    rmse_sin_prom = pivot_rmse.drop(columns='PROMEDIO')
    sns.heatmap(
        rmse_sin_prom.round(2),
        annot=True, fmt='.2f', cmap='YlOrRd',
        linewidths=0.5, ax=axes[0],
        cbar_kws={'label': 'RMSE (dólares)'}
    )
    axes[0].set_title('RMSE por Modelo y Acción', fontsize=11)
    axes[0].set_ylabel('Modelo')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].tick_params(axis='y', rotation=0)

    # Heatmap MAPE en porcentaje
    mape_pct = (pivot_mape.drop(columns='PROMEDIO') * 100)
    sns.heatmap(
        mape_pct.round(2),
        annot=True, fmt='.2f', cmap='YlOrRd',
        linewidths=0.5, ax=axes[1],
        cbar_kws={'label': 'MAPE (%)'}
    )
    axes[1].set_title('MAPE (%) por Modelo y Acción', fontsize=11)
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('images/grafica_2_heatmaps_rmse_mape.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("- grafica_2_heatmaps_rmse_mape.png")


graficar_heatmaps(pivot_rmse, pivot_mape)


# GRÁFICA 3: IMPACTO DE LA VALIDACIÓN CRUZADA (VC) EN CADA FAMILIA DE MODELOS

# Compara el RMSE promedio antes y después de optimizar hiperparámetros con VC.
# Cada grupo de barras representa una familia de modelos.
# Permite cuantificar cuánto mejora (o empeora) cada modelo al optimizarse.

def graficar_impacto_vc(pivot_rmse):
    """
    Barras agrupadas por familia de modelo mostrando RMSE sin VC y con VC.
    La diferencia entre barras indica el beneficio de la optimización.
    """
    # Pares de modelos: (sin VC, con VC)
    pares = [
        ('Ridge Lineal',          'Ridge Lineal con VC'),
        ('Ridge con Kernel RBF',  'Ridge con Kernel RBF y VC'),
        ('SVR Lineal',            'SVR Lineal con VC'),
        ('SVR con Kernel RBF',    'SVR con Kernel RBF y VC'),
    ]
    etiquetas  = ['Ridge Lineal', 'Ridge\nKernel RBF', 'SVR Lineal', 'SVR\nKernel RBF']
    x          = np.arange(len(pares))
    ancho      = 0.35

    rmse_sin = [pivot_rmse.loc[p[0], 'PROMEDIO'] for p in pares]
    rmse_con = [pivot_rmse.loc[p[1], 'PROMEDIO'] for p in pares]

    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - ancho / 2, rmse_sin, ancho,
                label='Sin optimización', color='#c0c0c0',
                edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + ancho / 2, rmse_con, ancho,
                label='Con validación cruzada (VC)', color='#2E86C1',
                edgecolor='black', linewidth=0.6)

    # Anotar la mejora o empeoramiento porcentual sobre la barra con VC
    for i, (s, c) in enumerate(zip(rmse_sin, rmse_con)):
        cambio = (s - c) / s * 100
        color_txt = '#1a7a1a' if cambio >= 0 else '#c0392b'
        signo    = '+' if cambio >= 0 else ''
        ax.text(x[i] + ancho / 2, c + 0.15,
                f'{signo}{cambio:.1f}%', ha='center', fontsize=9,
                color=color_txt, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=10)
    ax.set_ylabel('RMSE Promedio (dólares)', fontsize=11)
    ax.set_title('Impacto de la Validación Cruzada en Cada Familia de Modelos\n'
                 '(% = cambio en RMSE al optimizar parámetros)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/grafica_3_impacto_vc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("- grafica_3_impacto_vc.png")


graficar_impacto_vc(pivot_rmse)


# GRÁFICA 4: COMPLEJIDAD VS RMSE PROMEDIO

# Scatter plot que muestra el trade-off entre complejidad del modelo
# (número de parámetros) y su error de predicción.
# Permite identificar qué modelos ofrecen el mejor balance costo-beneficio.

def graficar_complejidad_vs_rmse(pivot_rmse, pivot_params):
    """
    Scatter donde cada punto es un modelo.
    Eje X: número promedio de parámetros (complejidad).
    Eje Y: RMSE promedio en el conjunto de prueba.
    """
    modelos_graf = [m for m in pivot_rmse.index if m != 'Random Walk']

    rmse_vals   = [pivot_rmse.loc[m, 'PROMEDIO']   for m in modelos_graf]
    params_vals = [pivot_params.loc[m, 'PROMEDIO']  for m in modelos_graf]
    colores_pts = [COLORES_MODELOS[m]               for m in modelos_graf]

    fig, ax = plt.subplots(figsize=(10, 6))

    for m, x, y, c in zip(modelos_graf, params_vals, rmse_vals, colores_pts):
        ax.scatter(x, y, color=c, s=120, zorder=3,
                   edgecolors='black', linewidth=0.7)
        # Offset de etiqueta para evitar solapamiento
        offset_x = 10
        offset_y = 0.05
        ax.annotate(m, xy=(x, y),
                    xytext=(x + offset_x, y + offset_y),
                    fontsize=8, arrowprops=None)

    # Línea horizontal de referencia: RMSE del Random Walk
    rmse_rw = pivot_rmse.loc['Random Walk', 'PROMEDIO']
    ax.axhline(rmse_rw, color=COLORES_MODELOS['Random Walk'],
               linestyle=':', linewidth=1.5,
               label=f'Random Walk (RMSE = {rmse_rw:.2f})')

    ax.set_xlabel('Número de Parámetros del Modelo (complejidad)', fontsize=11)
    ax.set_ylabel('RMSE Promedio en Prueba (dólares)', fontsize=11)
    ax.set_title('Complejidad del Modelo vs Error de Predicción\n', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/grafica_4_complejidad_vs_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("- grafica_4_complejidad_vs_rmse.png")


graficar_complejidad_vs_rmse(pivot_rmse, pivot_params)


# GRÁFICA 5: PREDICCIONES VS PRECIO REAL POR ACCIÓN

# Para cada acción se generan dos paneles:
#   Superior: modelos SIN validación cruzada + Random Walk vs precio real
#   Inferior: modelos CON validación cruzada + Random Walk vs precio real
#
# Mostrar los últimos N_DIAS del período de prueba para mayor claridad visual.

N_DIAS_GRAFICA = 100    # Días del conjunto de prueba a mostrar en la gráfica

MODELOS_SIN_VC = ['Random Walk', 'Ridge Lineal',
                  'Ridge con Kernel RBF', 'SVR Lineal', 'SVR con Kernel RBF']

MODELOS_CON_VC = ['Random Walk', 'Ridge Lineal con VC',
                  'Ridge con Kernel RBF y VC', 'SVR Lineal con VC',
                  'SVR con Kernel RBF y VC']


def graficar_predicciones_por_accion(ticker, resultados, y_test):
    """
    Dos paneles verticales para una acción:
    - Panel superior: modelos sin optimización vs precio real
    - Panel inferior: modelos con validación cruzada vs precio real
    Los últimos N_DIAS_GRAFICA días del período de prueba se grafican.
    """
    n    = min(N_DIAS_GRAFICA, len(y_test))
    dias = np.arange(n)
    real = y_test[-n:]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={'hspace': 0.45})
    fig.suptitle(f'Predicciones vs Precio Real — {ticker}\n'
                 f'(últimos {n} días del conjunto de prueba)',
                 fontsize=13, fontweight='bold')

    titulos = ['Modelos sin optimización de hiperparámetros',
               'Modelos con validación cruzada (parámetros optimizados)']

    for ax, grupo, titulo in zip(axes,
                                 [MODELOS_SIN_VC, MODELOS_CON_VC],
                                 titulos):
        # Precio real como referencia principal
        ax.plot(dias, real, color='black', linewidth=2,
                label='Precio Real', zorder=5)

        for nombre in grupo:
            if nombre not in resultados:
                continue
            pred = resultados[nombre]['preds'][-n:]
            rmse = resultados[nombre]['RMSE']
            ax.plot(dias, pred,
                    color=COLORES_MODELOS[nombre],
                    linestyle=ESTILOS_LINEA[nombre],
                    linewidth=1.3,
                    label=f'{nombre}  (RMSE={rmse:.2f})',
                    zorder=3)

        ax.set_title(titulo, fontsize=10, fontweight='bold')
        ax.set_xlabel('Días de prueba')
        ax.set_ylabel('Precio de cierre ($)')
        ax.legend(fontsize=7.5, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.savefig(f'images/grafica_5_predicciones_{ticker}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"- grafica_5_predicciones_{ticker}.png")


for ticker in ACCIONES:
    graficar_predicciones_por_accion(
        ticker,
        todos_resultados[ticker],
        todos_y_test[ticker]
    )


# GRÁFICA 6: RMSE POR ACCIÓN PARA CADA MODELO

# Permite ver cómo varía el error de cada modelo entre acciones.
# Útil para identificar qué acciones son más difíciles de predecir
# y si el ranking de modelos es consistente entre activos.

def graficar_rmse_por_accion(pivot_rmse):
    """
    Líneas conectadas: cada línea es un modelo y el eje X son las acciones.
    Muestra la variabilidad del error de cada modelo entre activos.
    """
    # Ordenamos los modelos por RMSE promedio para la leyenda
    orden = pivot_rmse['PROMEDIO'].sort_values().index.tolist()
    acciones_eje = [c for c in pivot_rmse.columns if c != 'PROMEDIO']

    fig, ax = plt.subplots(figsize=(11, 6))

    for modelo in orden:
        vals = [pivot_rmse.loc[modelo, acc] for acc in acciones_eje]
        ax.plot(acciones_eje, vals,
                color=COLORES_MODELOS[modelo],
                linestyle=ESTILOS_LINEA[modelo],
                marker='o', markersize=6, linewidth=1.6,
                label=f'{modelo}  (prom={pivot_rmse.loc[modelo, "PROMEDIO"]:.2f})')

    ax.set_xlabel('Acción', fontsize=11)
    ax.set_ylabel('RMSE (dólares)', fontsize=11)
    ax.set_title('RMSE por Acción en Cada Modelo\n',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/grafica_6_rmse_por_accion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("- grafica_6_rmse_por_accion.png")


graficar_rmse_por_accion(pivot_rmse)


# GRÁFICA 7: ESTÁTICO CON VC VS VENTANA DESLIZANTE

# Para cada familia de modelos, compara el RMSE promedio del modelo estático
# con el del modelo de ventana deslizante. Permite cuantificar el beneficio
# del reentrenamiento periódico siguiendo a Henrique et al. (2018).

MAPA_VD_EST = {
    'Ridge Lineal VD' : 'Ridge Lineal con VC',
    'Ridge RBF VD'    : 'Ridge con Kernel RBF y VC',
    'SVR Lineal VD'   : 'SVR Lineal con VC',
    'SVR RBF VD'      : 'SVR con Kernel RBF y VC',
}

ETIQUETAS_VD = ['Ridge Lineal', 'Ridge\nKernel RBF', 'SVR Lineal', 'SVR\nKernel RBF']

COLORES_VD = {
    'Estático (VC)' : '#2E86C1',
    'Ventana Deslizante' : '#E74C3C',
}

def graficar_estatico_vs_vd(pivot_rmse, pivot_rmse_vd):
    """
    Barras agrupadas: para cada familia de modelos muestra el RMSE promedio
    del modelo estático con VC y del modelo de ventana deslizante.
    """
    nombres_vd  = list(MAPA_VD_EST.keys())
    nombres_est = list(MAPA_VD_EST.values())
    x    = np.arange(len(nombres_vd))
    ancho = 0.35

    rmse_est = [pivot_rmse.loc[n, 'PROMEDIO']    for n in nombres_est]
    rmse_vd  = [pivot_rmse_vd.loc[n, 'PROMEDIO'] for n in nombres_vd]

    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - ancho / 2, rmse_est, ancho,
                label='Estático con VC', color=COLORES_VD['Estático (VC)'],
                edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + ancho / 2, rmse_vd,  ancho,
                label=f'Ventana Deslizante (W={VENTANA_DIAS})',
                color=COLORES_VD['Ventana Deslizante'],
                edgecolor='black', linewidth=0.6)

    for i, (e, v) in enumerate(zip(rmse_est, rmse_vd)):
        cambio  = (e - v) / e * 100
        color_t = '#1a7a1a' if cambio >= 0 else '#c0392b'
        signo   = '+' if cambio >= 0 else ''
        ax.text(x[i] + ancho / 2, v + 0.1,
                f'{signo}{cambio:.1f}%', ha='center', fontsize=9,
                color=color_t, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS_VD, fontsize=10)
    ax.set_ylabel('RMSE Promedio (dólares)', fontsize=11)
    ax.set_title(f'Estático con VC  vs  Ventana Deslizante (W = {VENTANA_DIAS} días)\n'
                 '(% = cambio de RMSE al usar ventana deslizante)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/grafica_7_estatico_vs_vd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("- grafica_7_estatico_vs_vd.png")


graficar_estatico_vs_vd(pivot_rmse, pivot_rmse_vd)


# GRÁFICA 8: PREDICCIONES ESTÁTICO vs VENTANA DESLIZANTE POR ACCIÓN
# Para el mejor modelo (SVR Lineal con VC / SVR Lineal VD), muestra
# cómo difieren las predicciones del modelo estático y el de ventana
# deslizante frente al precio real, tal como la Figura 2 del paper.

def graficar_comparacion_vd_por_accion(ticker, resultados_est, resultados_vd, n_dias=100):
    """
    Un solo panel con 3 curvas:
      - Precio real
      - Mejor modelo estático con VC (SVR Lineal)
      - Mejor modelo con ventana deslizante (SVR Lineal VD)
    """
    y_real  = resultados_vd['SVR Lineal VD']['y_test']
    pred_est = resultados_est['SVR Lineal con VC']['preds']
    pred_vd  = resultados_vd['SVR Lineal VD']['preds']

    n   = min(n_dias, len(y_real))
    dias = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dias, y_real[-n:],  color='black',  lw=2.0,  label='Precio Real', zorder=5)
    ax.plot(dias, pred_est[-n:], color='#2E86C1', lw=1.4,
            linestyle='--', label=f'SVR Lineal Estático  (RMSE={resultados_est["SVR Lineal con VC"]["RMSE"]:.2f})')
    ax.plot(dias, pred_vd[-n:],  color='#E74C3C', lw=1.4,
            linestyle=':',  label=f'SVR Lineal VD W={VENTANA_DIAS}  (RMSE={resultados_vd["SVR Lineal VD"]["RMSE"]:.2f})')

    ax.set_title(f'{ticker} — SVR Lineal: Estático vs Ventana Deslizante\n'
                 f'(últimos {n} días del conjunto de prueba)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Días de prueba')
    ax.set_ylabel('Precio de cierre ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'images/grafica_8_vd_{ticker}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"- grafica_8_vd_{ticker}.png")


for ticker in ACCIONES:
    graficar_comparacion_vd_por_accion(
        ticker,
        todos_resultados[ticker],
        todos_resultados_vd[ticker]
    )