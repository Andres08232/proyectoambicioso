# System Architecture

## Monorepo Structure

VISIONGOAT/
├── frontend/
├── backend/
├── docs/

## Backend Architecture

The backend follows a modular monolith architecture.

backend/
├── app/
│   ├── api/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── ingestion/
│   ├── features/
│   ├── ml/
│   ├── db/
│   ├── core/
│   └── utils/

## Architectural Principles

- API-first design
- modular architecture
- strong typing
- scalability
- maintainability
- minimal technical debt
- separation of concerns

## Data Flow

1. External football APIs
2. Data ingestion pipelines
3. PostgreSQL storage
4. Feature engineering
5. Prediction engine
6. Probability calibration
7. Value betting engine
8. Frontend dashboard

## Initial Data Sources

- Football-Data.org
- Understat
- Odds APIs

## Future Scaling

Future improvements may include:
- Redis caching
- asynchronous workers
- model versioning
- cloud object storage
- advanced ML pipelines