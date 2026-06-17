import pandas as pd
import os
import sys
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta solo estos paths
# ─────────────────────────────────────────────────────────────────────────────
XG_PATH   = r"D:\PROYECTO\data\raw\VisionGoat_Matches_xG.csv"
ODDS_PATH = r"D:\PROYECTO\data\raw\historico_cuotas.csv"
OUT_DIR   = r"D:\PROYECTO\data\processed"
 
# ─────────────────────────────────────────────────────────────────────────────
# MAPA DE NORMALIZACIÓN (bidireccional, exhaustivo)
# ─────────────────────────────────────────────────────────────────────────────
TEAM_ALIAS_MAP = {
    # Manchester United — todas las variantes conocidas
    "manchester utd":          "man utd",
    "man utd":                 "man utd",
    "manchester united":       "man utd",   # ← nuevo
    # Manchester City
    "manchester city":         "man city",  # ← nuevo
    "man city":                "man city",
    # Newcastle
    "newcastle utd":           "newcastle",
    "newcastle united":        "newcastle", # ← nuevo
    "newcastle":               "newcastle",
    # Sheffield
    "sheffield utd":           "sheffield",
    "sheffield united":        "sheffield", # ← nuevo
    "sheffield":               "sheffield",
    # Wolves
    "wolverhampton wanderers": "wolves",
    "wolves":                  "wolves",
    # Brighton
    "brighton and hove albion":"brighton",
    "brighton & hove albion":  "brighton",
    "brighton":                "brighton",
    # Tottenham
    "tottenham hotspur":       "tottenham",
    "tottenham":               "tottenham",
    # West Ham
    "west ham utd":            "west ham",
    "west ham united":         "west ham",  # ← nuevo
    "west ham":                "west ham",
    # Leicester
    "leicester city":          "leicester",
    "leicester":               "leicester",
    # Leeds
    "leeds utd":               "leeds",
    "leeds united":            "leeds",     # ← nuevo
    "leeds":                   "leeds",
    # Nottingham Forest — 4 variantes detectadas
    "nottingham forest":       "nottm forest",
    "nott'm forest":           "nottm forest",
    "nott'ham forest":         "nottm forest",  # ← nuevo (aparecía en faltantes)
    "nottm forest":            "nottm forest",
    # Luton
    "luton town":              "luton",
    "luton":                   "luton",
    # Bournemouth
    "afc bournemouth":         "bournemouth",
    "bournemouth":             "bournemouth",
    # Ipswich
    "ipswich town":            "ipswich",   # ← nuevo
    "ipswich":                 "ipswich",
    # Norwich
    "norwich city":            "norwich",   # ← nuevo
    "norwich":                 "norwich",
}
 
 
def normalize_team(name: str) -> str:
    """
    Normalización en 3 pasos:
      1. Lower + strip + quitar puntos
      2. Sustituir 'united' → 'utd' (cubre variantes no listadas)
      3. Lookup en alias map → si no está, devuelve el nombre ya limpio
    """
    s = str(name).lower().strip().replace(".", "").replace("'", "'")
    s = s.replace("united", "utd")
    return TEAM_ALIAS_MAP.get(s, s)
 
 
def load_and_prepare_xg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["HomeTeam_n"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeam_n"] = df["AwayTeam"].apply(normalize_team)
    assert df.duplicated(["Date", "HomeTeam_n", "AwayTeam_n"]).sum() == 0, \
        "¡El dataset xG tiene partidos duplicados! Revisa la fuente."
    return df
 
 
def load_and_prepare_odds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # dayfirst=True para formato DD/MM/YYYY del CSV de cuotas
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.normalize()
    df["HomeTeam_n"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeam_n"] = df["AwayTeam"].apply(normalize_team)
 
    # ── DEDUPLICAR ANTES DEL MERGE ──────────────────────────────────────────
    # El dataset de cuotas tiene 380 filas exactamente duplicadas.
    # Sin este paso el merge infla el xG a 2206 filas en vez de 1900.
    n_before = len(df)
    df = df.drop_duplicates(subset=["Date", "HomeTeam_n", "AwayTeam_n"], keep="first")
    n_after = len(df)
    if n_before != n_after:
        print(f"  [INFO] Cuotas deduplicadas: {n_before} → {n_after} filas "
              f"({n_before - n_after} duplicados eliminados)")
    return df
 
 
def merge_datasets(df_xg: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    odds_cols = ["Date", "HomeTeam_n", "AwayTeam_n", "B365H", "B365D", "B365A"]
    df = pd.merge(
        df_xg,
        df_odds[odds_cols],
        on=["Date", "HomeTeam_n", "AwayTeam_n"],
        how="left"
    )
    # Garantía de integridad: nunca más filas que xG
    assert len(df) == len(df_xg), (
        f"¡El merge infló el dataset! xG={len(df_xg)}, merged={len(df)}. "
        "Hay claves duplicadas en las cuotas tras deduplicar. Investiga."
    )
    return df
 
 
def classify_failures(df: pd.DataFrame, odds_start: pd.Timestamp, odds_end: pd.Timestamp) -> pd.DataFrame:
    """
    Asigna una etiqueta de auditoría a cada fila:
      ✅ matched            → cuota encontrada
      ❌ out_of_odds_range  → fecha fuera de cobertura del CSV de cuotas
      ❌ no_odds_found      → dentro del rango pero sin cuota (equipo nuevo, etc.)
    """
    conditions = [
        df["B365H"].notnull(),
        df["Date"] < odds_start,
        df["Date"] > odds_end,
    ]
    choices = ["matched", "out_of_odds_range", "out_of_odds_range"]
    df["merge_status"] = pd.Series(
        pd.np.select(conditions, choices, default="no_odds_found")
        if hasattr(pd, "np")
        else __import__("numpy").select(conditions, choices, default="no_odds_found"),
        index=df.index
    )
    return df
 
 
def build_audit_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un CSV de auditoría solo para los no-matched,
    con columna 'fail_reason' descriptiva.
    """
    failed = df[df["merge_status"] != "matched"].copy()
    failed["fail_reason"] = failed["merge_status"].map({
        "out_of_odds_range": "Fuera del rango de fechas del CSV de cuotas (temporada 20/21 sin cobertura)",
        "no_odds_found":     "Dentro del rango pero sin cuota encontrada (equipo/alias no mapeado o partido extra)"
    })
    return failed[["Date", "HomeTeam", "AwayTeam", "HomeTeam_n", "AwayTeam_n",
                   "FTHG", "FTAG", "merge_status", "fail_reason"]]
 
 
def print_summary(df: pd.DataFrame):
    total      = len(df)
    matched    = (df["merge_status"] == "matched").sum()
    oor        = (df["merge_status"] == "out_of_odds_range").sum()
    no_odds    = (df["merge_status"] == "no_odds_found").sum()
 
    print("\n" + "═" * 60)
    print("  RESUMEN DEL MERGE")
    print("═" * 60)
    print(f"  Total partidos xG    : {total:>6}")
    print(f"  ✅ Matched           : {matched:>6}  ({matched/total*100:.1f}%)")
    print(f"  ❌ Fuera de rango    : {oor:>6}  ({oor/total*100:.1f}%)  → sin cobertura en cuotas")
    print(f"  ❌ No encontrado     : {no_odds:>6}  ({no_odds/total*100:.1f}%)  → revisar audit.csv")
    print(f"  🔒 Integridad        : {total} filas = {len(df)} filas  OK")
    print("═" * 60)
 
 
def build_master_dataset(
    xg_path: str = XG_PATH,
    odds_path: str = ODDS_PATH,
    out_dir: str = OUT_DIR,
):
    os.makedirs(out_dir, exist_ok=True)
 
    print("\n[1/5] Cargando xG dataset...")
    df_xg = load_and_prepare_xg(xg_path)
    print(f"      {len(df_xg)} partidos | {df_xg['Date'].min().date()} → {df_xg['Date'].max().date()}")
 
    print("[2/5] Cargando y deduplicando cuotas...")
    df_odds = load_and_prepare_odds(odds_path)
    odds_start = df_odds["Date"].min()
    odds_end   = df_odds["Date"].max()
    print(f"      {len(df_odds)} partidos únicos | {odds_start.date()} → {odds_end.date()}")
 
    print("[3/5] Mergeando...")
    df_merged = merge_datasets(df_xg, df_odds)
 
    print("[4/5] Clasificando resultados...")
    import numpy as np
    conditions = [
        df_merged["B365H"].notnull(),
        df_merged["Date"] < odds_start,
        df_merged["Date"] > odds_end,
    ]
    choices = ["matched", "out_of_odds_range", "out_of_odds_range"]
    df_merged["merge_status"] = np.select(conditions, choices, default="no_odds_found")
 
    print_summary(df_merged)
 
    print("[5/5] Guardando archivos...")
 
    # Dataset maestro completo (sin columnas auxiliares de normalización)
    master_path = os.path.join(out_dir, "VisionGoat_Master_Dataset.csv")
    df_export = df_merged.drop(columns=["HomeTeam_n", "AwayTeam_n"])
    df_export.to_csv(master_path, index=False)
    print(f"      ✅ Master dataset   → {master_path}")
 
    # Auditoría de no-matched
    audit = build_audit_report(df_merged)
    audit_path = os.path.join(out_dir, "audit_no_match.csv")
    audit.to_csv(audit_path, index=False)
    print(f"      📋 Audit report     → {audit_path}  ({len(audit)} filas)")
 
    # Solo los matcheados (para modelos que requieran cuotas)
    matched_path = os.path.join(out_dir, "VisionGoat_Master_WithOdds.csv")
    df_export[df_export["merge_status"] == "matched"].drop(columns=["merge_status"]).to_csv(
        matched_path, index=False
    )
    print(f"      📊 Sólo matcheados  → {matched_path}")
 
    return df_merged
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MODO STANDALONE — también importable como módulo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Permite override de paths por argumentos de línea de comandos:
    # python visiongoat_merge_engine.py [xg_path] [odds_path] [out_dir]
    args = sys.argv[1:]
    kwargs = {}
    if len(args) >= 1: kwargs["xg_path"]  = args[0]
    if len(args) >= 2: kwargs["odds_path"] = args[1]
    if len(args) >= 3: kwargs["out_dir"]   = args[2]
    build_master_dataset(**kwargs)