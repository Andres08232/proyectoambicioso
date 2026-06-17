# API Contracts

Este documento define el lenguaje y la estructura de comunicación estricta entre el frontend y el backend de VisionGoat.

## GET /matches

Returns upcoming and recent football matches.

Example response:
```json
[
  {
    "id": 1,
    "home_team": "Arsenal",
    "away_team": "Liverpool",
    "match_date": "2026-05-30T19:00:00Z"
  }
]
```

## GET /predictions

Returns model probabilities.

Example response:
```json
[
  {
    "match_id": 1,
    "home_win_prob": 0.48,
    "draw_prob": 0.27,
    "away_win_prob": 0.25
  }
]
```

## GET /value-bets

Returns bets with positive expected value.

Example response:
```json
[
  {
    "match_id": 1,
    "market": "over_2_5",
    "bookmaker_odds": 2.1,
    "model_probability": 0.57,
    "expected_value": 0.197
  }
]
```

## GET /metrics

Returns model performance metrics. 

Core Metrics:
- ROI
- CLV
- Brier Score
- Log Loss