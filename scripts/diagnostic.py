import pandas as pd

xg_df = pd.read_csv('data/raw/VisionGoat_Matches_xG.csv')
odds_df = pd.read_csv('data/raw/PremierLeague26England.csv')

# Convertir fechas explícitamente
xg_df['Date'] = pd.to_datetime(xg_df['Date'])
odds_df['Date'] = pd.to_datetime(odds_df['Date'], dayfirst=True) # Probamos dayfirst=True

print(f"Primeras fechas xG: {xg_df['Date'].iloc[0]}")
print(f"Primeras fechas odds: {odds_df['Date'].iloc[0]}")

# Ver si los equipos coinciden
print(f"Equipos xG (muestra): {xg_df['HomeTeam'].unique()[:5]}")
print(f"Equipos Odds (muestra): {odds_df['HomeTeam'].unique()[:5]}")