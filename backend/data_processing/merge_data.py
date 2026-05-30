import pandas as pd
import os

def merge_datasets():
    # 1. Cargar datasets
    # Ajusta estas rutas según donde tengas tus archivos
    xg_df = pd.read_csv('data/raw/VisionGoat_Matches_xG.csv')
    odds_df = pd.read_csv('data/raw/PremierLeague26England.csv') 

    # 2. Asegurar formatos de fecha
    xg_df['Date'] = pd.to_datetime(xg_df['Date'])
    odds_df['Date'] = pd.to_datetime(odds_df['Date'])

    # 3. Merge (Unión) por Date, HomeTeam, AwayTeam
    # Usamos 'left' para mantener todas las filas de nuestro archivo de xG
    merged = pd.merge(xg_df, odds_df[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']], 
                      on=['Date', 'HomeTeam', 'AwayTeam'], how='left')

    # 4. Guardar resultado
    merged.to_csv('data/raw/VisionGoat_Full_Data.csv', index=False)
    print(f"Merge exitoso. Filas totales: {len(merged)}")
    print(f"Filas con cuotas encontradas: {merged['B365H'].notnull().sum()}")

if __name__ == "__main__":
    merge_datasets()