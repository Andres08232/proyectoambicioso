# (Nombre en Progreso): Motor de Análisis Predictivo para Apuestas Deportivas

## 🎯 Objetivo
(Nombre en Progreso) es un motor analítico diseñado para identificar ineficiencias en el mercado de apuestas deportivas (**Value Betting**). Nuestro objetivo no es "adivinar" resultados, sino construir un modelo estadístico capaz de detectar cuando la probabilidad real de un evento es mayor a la cuota ofrecida por las casas de apuestas.

## 🧠 Filosofía del Proyecto
La mayoría de los sistemas de apuestas fallan por ser "Naive" (simples). (Nombre en Progreso) se basa en tres pilares fundamentales:

1.  **Detección de Valor:** Nuestra métrica de éxito no es acertar el ganador, sino la consistencia del ROI (Retorno de Inversión) al identificar apuestas con valor positivo.
2.  **Sistema Elo Dinámico:** Utilizamos un motor de fuerza relativa (Elo) que aprende constantemente. A diferencia de un promedio simple, este sistema pondera el valor de las victorias según la fuerza del oponente y el margen de goles.
3.  **Arquitectura Modular (Config-Driven):** Hemos diseñado el sistema para ser agnóstico a la liga. El motor lógico es universal; las ligas son configuraciones que inyectamos al sistema. Esto nos permite escalar a cualquier liga del mundo sin duplicar código.

## ⚙️ Arquitectura del Sistema
- **`PredictionEngine` (El Cerebro):** Implementación profesional orientada a objetos que procesa el historial de partidos (walk-forward) y calcula probabilidades mediante Elo Rating.
- **Configuración (Modular):** Parámetros independientes por liga (`k_factor`, `home_advantage_elo`, etc.).
- **Pipeline de Datos:** Estructura para procesar archivos CSV de cualquier liga de forma estandarizada.
- **Optimizador (Grid Search):** Herramienta que calibra automáticamente nuestro motor a la realidad estadística de cada liga específica.

## 📈 Estado Actual del Modelo (Premier League)
Hemos completado una fase intensiva de calibración:
- **Estado:** Motor calibrado y optimizado.
- **ROI:** Hemos reducido la varianza significativamente, pasando de -12% a -6.25% tras optimizar los hiper-parámetros (k-factor y ventaja de localía).
- **Conclusión:** El motor es robusto. La brecha restante del 6% requiere añadir "contexto" externo (formación, rachas, bajas) para pasar a terreno positivo.

## 🚀 Hoja de Ruta (Roadmap)
- [x] **Fase 1:** Script básico de detección.
- [x] **Fase 2:** Arquitectura modular y sistema Elo estándar.
- [x] **Fase 3:** Optimización mediante Grid Search (parámetros calibrados).
- [x] **Fase 4:** Elo ajustado por Goles (diferenciando dominio vs. suerte).
- [ ] **Fase 5 (Siguiente):** Feature Engineering (Añadir rachas de forma y variables de contexto).
- [ ] **Fase 6:** Preparación para el Mundial.

## 🛠️ ¿Cómo podemos colaborar?
Este proyecto es un entorno de desarrollo activo. Si tienes ideas o quieres validar datos:
- **Análisis de Variables:** ¿Qué factores crees que influyen más en el desempeño de un equipo (lesiones, clima, calendario)? Podemos añadirlo como una nueva "feature".
- **QA/Validación:** Puedes tomar los archivos CSV de nuevas ligas y correr el optimizador para ver si el comportamiento de los parámetros es similar al de la Premier League.
- **Reportes:** Revisa los commits en GitHub para ver el progreso detallado de las funcionalidades.
