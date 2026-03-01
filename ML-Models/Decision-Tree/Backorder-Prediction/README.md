
## Model Explainability

SHAP (SHapley Additive exPlanations) is used to interpret model predictions.

Benefits:
- Identify features influencing backorders
- Improve supply chain decision making
- Increase trust in ML predictions

Visualization:
- SHAP summary plot
- Feature impact analysis


| Skill            | Level |
| ---------------- | ----- |
| Data Science     | ⭐⭐⭐⭐  |
| Machine Learning | ⭐⭐⭐⭐  |
| Explainable AI   | ⭐⭐⭐⭐  |
| ML Pipeline      | ⭐⭐⭐⭐  |
| Model Deployment | ⭐⭐⭐⭐  |
| API Development  | ⭐⭐⭐⭐  |




## Deployment

This project exposes the trained model using FastAPI.

Run API:

uvicorn api.main:app --reload

API Docs:

http://127.0.0.1:8000/docs


model (Decision Tree / XGBoost) typically predicts high probability of backorder when

| Feature                    | Risk Pattern              |
| -------------------------- | ------------------------- |
| inventory (`national_inv`) | **very low**              |
| forecast                   | **very high**             |
| sales                      | **high**                  |
| lead_time                  | **long**                  |
| in_transit_qty             | **low**                   |
| min_bank                   | **higher than inventory** |
| pieces_past_due            | **> 0**                   |
| local_bo_qty               | **> 0**                   |




For 95% Probability
{
"national_inv": 0,
"lead_time": 30,
"in_transit_qty": 0,
"forecast_3_month": 500,
"forecast_6_month": 800,
"forecast_9_month": 1200,
"sales_1_month": 120,
"sales_3_month": 350,
"sales_6_month": 600,
"sales_9_month": 900,
"min_bank": 100,
"pieces_past_due": 20,
"perf_6_month_avg": 0.4,
"perf_12_month_avg": 0.45,
"local_bo_qty": 10
}

{
  "prediction": true,
  "probability": 0.9588819875776398
}
HTTP status: 200