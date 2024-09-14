import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Función para cargar y preparar los datos
def cargar_datos(file_path):
    df = pd.read_excel(file_path, sheet_name='Tipos_de_cultivo_Palmira')
    
    # Asegurarse de que las columnas numéricas no contengan texto o NaN
    def convertir_a_numerico(df, columnas):
        for columna in columnas:
            df[columna] = pd.to_numeric(df[columna], errors='coerce')  # Convertir a numérico, NaN si hay error
            if df[columna].isnull().sum() > 0:
                print(f"Advertencia: La columna '{columna}' tiene valores no numéricos o nulos, que serán eliminados.")
        df.dropna(subset=columnas, inplace=True)  # Eliminar filas con valores NaN en las columnas indicadas
        return df

    # Especificar las columnas que deben ser numéricas
    columnas_numericas = ['Superficie Plantada (Hectáreas)', 'Rendimientos Promedio (Toneladas/Hectárea)', 'Producción (Toneladas)']
    
    # Convertir a numérico y eliminar filas problemáticas
    df = convertir_a_numerico(df, columnas_numericas)
    return df

# 1. Análisis Descriptivo
def analisis_descriptivo(df):
    print("Resumen estadístico:\n", df.describe())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Año', y='Superficie Plantada (Hectáreas)', hue='Tipo de cultivo', data=df)
    plt.title('Superficie Plantada por Año y Tipo de Cultivo')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Año', y='Rendimientos Promedio (Toneladas/Hectárea)', hue='Tipo de cultivo', data=df)
    plt.title('Tendencia de Rendimientos Promedio por Año y Tipo de Cultivo')
    plt.show()

# 2. Análisis Diagnóstico
def analisis_diagnostico(df):
    if df.empty:
        print("Error: El DataFrame está vacío. No se puede realizar el análisis.")
        return None, None, None
    
    correlacion = df[['Superficie Plantada (Hectáreas)', 'Rendimientos Promedio (Toneladas/Hectárea)', 'Producción (Toneladas)']].corr()
    print("Correlación entre las variables:\n", correlacion)

    # Regresión lineal: Rendimientos vs Superficie Plantada
    X_rendimiento = df[['Superficie Plantada (Hectáreas)']]
    Y_rendimiento = df['Rendimientos Promedio (Toneladas/Hectárea)']
    X_rendimiento = sm.add_constant(X_rendimiento)
    modelo_rendimiento = sm.OLS(Y_rendimiento, X_rendimiento).fit()
    print("\nResumen de la regresión de Rendimiento vs Superficie Plantada:\n", modelo_rendimiento.summary())

    # Regresión lineal: Producción vs Superficie Plantada
    X_produccion = df[['Superficie Plantada (Hectáreas)']]
    Y_produccion = df['Producción (Toneladas)']
    X_produccion = sm.add_constant(X_produccion)
    modelo_produccion = sm.OLS(Y_produccion, X_produccion).fit()
    print("\nResumen de la regresión de Producción vs Superficie Plantada:\n", modelo_produccion.summary())

    # Regresión lineal múltiple: Producción vs Superficie Plantada y Rendimientos
    X_multiple = df[['Superficie Plantada (Hectáreas)', 'Rendimientos Promedio (Toneladas/Hectárea)']]
    Y_multiple = df['Producción (Toneladas)']
    X_multiple = sm.add_constant(X_multiple)
    modelo_multiple = sm.OLS(Y_multiple, X_multiple).fit()
    print("\nResumen de la regresión múltiple de Producción vs Superficie Plantada y Rendimientos:\n", modelo_multiple.summary())

    return modelo_rendimiento, modelo_produccion, modelo_multiple
# 2. Análisis Prescrip
def analisis_prescriptivo(modelo_rendimiento, modelo_produccion, modelo_multiple):
    if modelo_rendimiento is None or modelo_produccion is None or modelo_multiple is None:
        print("Error: Los modelos no se han generado. No se puede realizar el análisis prescriptivo.")
        return
    
    superficies = [100, 200, 300, 400, 500]
    rendimientos_esperados = []
    produccion_esperada = []

    for superficie in superficies:
        # Predicción del rendimiento
        X_nueva_rendimiento = pd.DataFrame({'Superficie Plantada (Hectáreas)': [superficie]})
        X_nueva_rendimiento = sm.add_constant(X_nueva_rendimiento, has_constant='add')
        prediccion_rendimiento = modelo_rendimiento.predict(X_nueva_rendimiento)
        
        # Predicción de la producción usando los rendimientos predichos
        rendimiento_estimado = prediccion_rendimiento[0]
        X_nueva_produccion = pd.DataFrame({
            'Superficie Plantada (Hectáreas)': [superficie],
            'Rendimientos Promedio (Toneladas/Hectárea)': [rendimiento_estimado]
        })
        X_nueva_produccion = sm.add_constant(X_nueva_produccion, has_constant='add')
        prediccion_produccion = modelo_multiple.predict(X_nueva_produccion)
        
        rendimientos_esperados.append(rendimiento_estimado)
        produccion_esperada.append(prediccion_produccion[0])

    # Gráfico de simulación de rendimiento
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=superficies, y=rendimientos_esperados, marker='o', label='Rendimiento Esperado')
    plt.title('Simulación: Superficie Plantada vs Rendimiento Esperado')
    plt.xlabel('Superficie Plantada (Hectáreas)')
    plt.ylabel('Rendimiento Esperado (Toneladas/Hectárea)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Gráfico de simulación de producción
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=superficies, y=produccion_esperada, marker='o', label='Producción Esperada')
    plt.title('Simulación: Superficie Plantada vs Producción Esperada')
    plt.xlabel('Superficie Plantada (Hectáreas)')
    plt.ylabel('Producción Esperada (Toneladas)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Recomendaciones basadas en el análisis
    print("Recomendaciones basadas en el análisis:")
    for i, superficie in enumerate(superficies):
        print(f"Para una superficie plantada de {superficie} hectáreas, se espera un rendimiento de {rendimientos_esperados[i]:.2f} toneladas/hectárea y una producción de {produccion_esperada[i]:.2f} toneladas.")

# Ejemplo de uso del código
if __name__ == "__main__":
    # 1. Cargar datos
    file_path = r"C:\Users\M_arl\python\Tipos_de_cultivo_Palmira.xlsx"  # Cambia la ruta por la ubicación correcta de tu archivo
    df = cargar_datos(file_path)
    
    # 2. Análisis Descriptivo
    print("Análisis Descriptivo:")
    analisis_descriptivo(df)
    
    # 3. Análisis Diagnóstico
    print("Análisis Diagnóstico:")
    modelo_rendimiento, modelo_produccion, modelo_multiple = analisis_diagnostico(df)
    
    # 4. Análisis Prescriptivo
    print("Análisis Prescriptivo:")
    analisis_prescriptivo(modelo_rendimiento, modelo_produccion, modelo_multiple)
