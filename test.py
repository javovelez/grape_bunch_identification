import os
import pandas as pd

# Ruta de la carpeta que contiene los archivos CSV
carpeta = 'F:/Escritorio/repo_2023/identificaci-nDeRacimos/output/2023.03_captura_2/180_prueba_radios/thresh_07_igual_id/'

# Lista para almacenar los DataFrames de cada archivo CSV
lista_df = []

# Bucle para iterar a través de cada archivo CSV en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta, archivo)
        df = pd.read_csv(ruta_archivo, index_col=0, header=0)
        lista_df.append(df)

# Combinar todos los DataFrames en uno solo
df_final = pd.concat(lista_df, ignore_index=True)

# Mostrar el DataFrame final
print(df_final)