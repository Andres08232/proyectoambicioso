import pandas as pd

# 1. Cargar el dataset que acabas de subir
df = pd.read_csv('D:\\PROYECTO\\data\\raw\\final_matches.csv')

# 2. Filtrar solo las filas donde el equipo jugó de local ('Home')
# Como cada partido tiene dos filas, al tomar solo las 'Home' obtenemos un registro único por partido.
df_home = df[df['venue'] == 'Home'].copy()

# 3. Renombrar las columnas para que coincidan con tu formato clásico y asignar el xG
df_clean = df_home.rename(columns={
    'date': 'Date',
    'team': 'HomeTeam',
    'opponent': 'AwayTeam',
    'gf': 'FTHG',  # Full Time Home Goals
    'ga': 'FTAG',  # Full Time Away Goals
    'xg': 'Home_xG',
    'xga': 'Away_xG'
})

# 4. Seleccionar solo las columnas que importan para el modelo Elo
columnas_finales = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Home_xG', 'Away_xG']
df_final = df_clean[columnas_finales]

# 5. Guardar el nuevo dataset unificado
df_final.to_csv('VisionGoat_Matches_xG.csv', index=False)
print("¡Dataset procesado y listo para el backtest!")