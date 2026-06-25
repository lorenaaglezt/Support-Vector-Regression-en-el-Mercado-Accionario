# ANÁLISIS DE VOLATILIDAD DE ACCIONES
# Herramienta complementaria para justificación de desempeño de modelos predictivos

import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def calcular_metricas_volatilidad(tickers, benchmark="^GSPC", start="2021-01-04", end="2026-05-02"):
    """
    Calcula la volatilidad histórica, Beta y el rango diario promedio
    para una lista de acciones, comparadas contra el índice S&P 500.
    """
    print(f"\nDescargando datos del mercado (Benchmark: {benchmark})...")
    
    # Descargar el índice de referencia (S&P 500)
    df_mercado = yf.download(benchmark, start=start, end=end, progress=False)
    if isinstance(df_mercado.columns, pd.MultiIndex):
        df_mercado.columns = df_mercado.columns.droplevel(1)
        
    # Rendimientos diarios del mercado
    retornos_mercado = df_mercado['Close'].pct_change().dropna()
    varianza_mercado = retornos_mercado.var()

    resultados = []

    for ticker in tickers:
        print(f"Analizando volatilidad de {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Manejo seguro por si falla la descarga de Yahoo Finance
        if df.empty:
            print(f"  [!] Advertencia: No se pudieron descargar datos de {ticker}.")
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Calcular rendimientos diarios
        df['Rendimiento'] = df['Close'].pct_change()
        retornos_accion = df['Rendimiento'].dropna()

        # A) Volatilidad Histórica Anualizada (Desviación estándar)
        # Se multiplica por la raíz cuadrada de los días bursátiles (aprox 252)
        volatilidad_anual = retornos_accion.std() * np.sqrt(252)

        # B) Beta (Riesgo sistemático frente al mercado)
        datos_alineados = pd.concat([retornos_accion, retornos_mercado], axis=1, join='inner')
        datos_alineados.columns = ['Accion', 'Mercado']
        covarianza = datos_alineados.cov().iloc[0, 1]
        beta = covarianza / varianza_mercado

        # C) Rango Diario Promedio (High - Low) expresado en porcentaje
        rango_diario_pct = ((df['High'] - df['Low']) / df['Low']).mean()

        resultados.append({
            'Acción': ticker,
            'Vol_Anualizada': volatilidad_anual * 100,
            'Beta': beta,
            'Rango_Diario': rango_diario_pct * 100
        })

    # Formatear resultados en un DataFrame
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by='Vol_Anualizada', ascending=False)
    
    return df_resultados


if __name__ == "__main__":
    # Emisoras a comparar
    acciones_a_medir = ['HPE', 'AVGO', 'CSCO', 'DELL', 'GOOGL', 'MU']

    df_volatilidad = calcular_metricas_volatilidad(
        tickers=acciones_a_medir, 
        start='2021-01-01',
        end='2026-05-01'
    )

    print("\nREPORTE DE VOLATILIDAD DE ACCIONES (2021 - 2026)\n")
    
    # Formato seguro para impresión limpia en consola usando formatters
    formato_impresion = {
        'Vol_Anualizada': lambda x: f"{x:.2f}%",
        'Beta': lambda x: f"{x:.3f}",
        'Rango_Diario': lambda x: f"{x:.2f}%"
    }

    print(df_volatilidad.to_string(index=False, formatters=formato_impresion))
    print("Nota: Una volatilidad alta suele limitar la capacidad de los")
    print("modelos de ML para minimizar el RMSE debido al ruido estocástico.")