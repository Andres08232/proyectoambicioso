<<<<<<< HEAD
=======
[README.md](https://github.com/user-attachments/files/28374489/README.md)
>>>>>>> ba50f3d38cda50421a1fcb71d0efa5174147dbdc
# VisionGoat: Sistema de Análisis de Valor en Apuestas

## 🎯 Objetivo
VisionGoat es un motor analítico diseñado para identificar ineficiencias en el mercado de apuestas deportivas (Value Betting). Nuestro objetivo no es "adivinar" el ganador, sino encontrar situaciones donde la probabilidad real de un evento es mayor a la que dicta la cuota de la casa de apuestas.

## 🧠 Lógica detrás del Proyecto
La mayoría de los sistemas fallan porque usan modelos "Naive" (simples). VisionGoat se basa en tres pilares:
1. **Detección de Valor:** Apostamos solo cuando: `(Probabilidad del Modelo * Cuota) > 1`.
2. **Sistema Elo:** Utilizamos un motor dinámico (Elo Rating) que ajusta la fuerza de cada equipo basándose en sus resultados anteriores. A diferencia de un promedio simple, el Elo da más peso a las victorias contra rivales fuertes.
3. **Arquitectura Modular:** El motor es universal (Elo). Las ligas (Premier, Bundesliga, etc.) son configuraciones que se inyectan al sistema. Esto nos permite escalar a cualquier liga sin cambiar el código core.

## ⚙️ Estructura del Sistema
- `PredictionEngine`: El cerebro matemático que calcula las probabilidades (Walk-forward).
- `Configuración`: Parámetros específicos por liga (`k_factor`, `home_advantage_elo`).
- `Backtesting`: Pipeline para probar nuestra estrategia con datos históricos y validar el ROI.

## 🚀 Hoja de Ruta (Roadmap)
- [x] **Fase 1:** Script básico de detección.
- [x] **Fase 2:** Arquitectura modular y sistema Elo estándar.
- [x] **Fase 3:** Optimización mediante Grid Search.
- [ ] **Fase 4 (En curso):** Elo ajustado por Goles (para diferenciar dominio vs. suerte).
- [ ] **Fase 5:** Implementación para el Mundial.

## 🛠️ Cómo colaborar
- Si añades una liga, solo necesitas un archivo CSV y añadir su configuración en `config.py`.
<<<<<<< HEAD
- Antes de subir cambios, asegúrate de correr los tests básicos y revisar el ROI del backtest.
=======
- Antes de subir cambios, asegúrate de correr los tests básicos y revisar el ROI del backtest.
>>>>>>> ba50f3d38cda50421a1fcb71d0efa5174147dbdc
