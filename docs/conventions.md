# Development Conventions

## Git Branches

Feature branches:

- feat/backend-foundation
- feat/elo-system
- feat/prediction-engine

Bugfix branches:

- fix/api-timeout
- fix/db-connection

## Commit Convention

Format:

type(scope): message

Examples:

- feat(api): add matches endpoint
- fix(db): resolve postgres connection issue
- refactor(ml): simplify elo calculations

## Python Style

- type hints required
- async where appropriate
- modular services
- no business logic in routes

## API Conventions

- RESTful naming
- JSON responses
- snake_case backend
- camelCase frontend mapping allowed

## Database

- PostgreSQL only
- Alembic migrations required
- no raw SQL unless necessary

## General Rules

- avoid overengineering
- keep modules decoupled
- prioritize readability
- prioritize maintainability
- write production-oriented code