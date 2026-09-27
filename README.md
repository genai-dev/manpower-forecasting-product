Manpower Forecasting Product
Overview
This project is an ML-based manpower forecasting solution designed to forecast workforce metrics for the next financial year using historical monthly manpower data.
The current forecasting horizon is:
- Historical period: Apr-2023 to Mar-2026
- Forecast period: Apr-2026 to Mar-2027
The solution works primarily at Channel × Zone × Month level and is intended to support business planning by converting historical manpower movement into forward-looking workforce estimates.
Business Use Case
The objective is to estimate future manpower requirements and workforce movement for each business segment.
The main metrics are:
- Closing FLS Count
- Exit Count
- Opening FLS Count
- Hiring Count
- Attrition Rate
Instead of manually preparing multiple Excel pivots and calculations for every month, channel and zone, the pipeline standardizes the raw data, creates historical monthly features, trains ML models and generates the next 12 months of forecasts.
ML Approach
The project uses a Random Forest Regressor for forecasting the primary manpower targets.
Why Random Forest?
Random Forest is suitable for this use case because it:
- Handles non-linear relationships well.
- Works with multiple categorical and numerical features after preprocessing.
- Can learn interactions between Channel, Zone, Month and historical manpower patterns.
- Is less sensitive to outliers than a single decision tree.
- Does not require the strict statistical assumptions of traditional linear models.
The Random Forest consists of multiple decision trees. Each tree learns different patterns from the historical data, and the final regression prediction is obtained by combining the predictions from all trees.
Forecasting Strategy
The forecasting process is divided into two parts.
1. ML-predicted metrics
The model forecasts:
- Closing FLS Count
- Exit Count
These forecasts are generated month-wise for the next financial year.
2. Derived business metrics
The remaining metrics are calculated using business logic:
Opening FLS(t) = Closing FLS(t-1)

Hiring(t) = Closing FLS(t) - Opening FLS(t) + Exit(t)

Attrition Rate(t) = Exit(t) / Opening FLS(t)
This keeps the dependent manpower metrics internally consistent instead of training separate models for every value.
Data Preparation
Historical source data is cleaned and standardized before model training.
Zone Standardization
Zones are normalized to only:
- East
- West
- North
- South
Examples:
East 1 / East 2 / East 3 / East 4   -> East
West 1 / West 2 / West 3 / West 4   -> West
North 1 / North 2 / North 3 / North 4 -> North
South 1 / South 2 / South 3 / South 4 -> South
Location-based mappings used in preprocessing include:
Location	Zone
Kolkata	East
Delhi	North
Guwahati	East
Siliguri	East
Burdwan	East
Patna	East
Raipur	East
West	West


Channel Filtering
Channels containing the following terms are excluded from the modelling scope:
- Credit
- Group
Channel Standardization
Different source labels are mapped to common business channels where required.
Examples:
Source / Vertical	Standard Channel
IB Others	IB Others
Policy Bazaar	Digital Partners
JSFB	JSFB
Bandhan Bank	Bandhan Bank
AXIS BRO / TELLER	Axis Bank
AXIS LIABILITY SALES	Axis Bank


Channel mapping is applied only where source labels require standardization; valid existing channel names are retained.
Model Features
The model can use historical combinations such as:
- Month
- Financial year / time sequence
- Channel
- Zone
- Historical Closing FLS
- Historical Exit Count
- Lag / historical manpower behaviour
The architecture is designed so additional business variables can be incorporated as the forecasting model evolves, for example:
- Seasonality
- NOP
- FLS Productivity
- Sales-related variables
- Other approved business drivers
Forecasting Flow
Raw Historical Files
        |
        v
Data Cleaning & Standardization
        |
        +--> Channel filtering / mapping
        |
        +--> Zone normalization
        |
        v
Monthly Historical Dataset
        |
        v
Feature Engineering
        |
        v
Random Forest Regression
        |
        +--> Closing FLS Forecast
        |
        +--> Exit Forecast
        |
        v
Business Formula Layer
        |
        +--> Opening FLS
        +--> Hiring
        +--> Attrition Rate
        |
        v
Apr-2026 to Mar-2027 Forecast Output
Expected Output
The final forecast is expected to provide month-wise manpower information by business segment, including:
Month	Channel	Zone	Opening FLS	Closing FLS	Exit	Hiring	Attrition Rate


The output can then be consumed by business planning teams or extended into reporting / visualization layers.
Key Design Principle
The solution does not simply take a three-year arithmetic average.
Random Forest learns patterns from historical records and available features. The model predicts the primary future manpower values, while dependent metrics are calculated using defined manpower equations.
This gives a more structured and scalable approach than manually forecasting every Channel × Zone × Month combination in Excel.
Current Scope
The current version focuses on creating a reusable end-to-end forecasting pipeline for manpower planning.
The broader roadmap can include:
- Additional business variables.
- Model comparison and back-testing.
- Hyperparameter tuning.
- Forecast accuracy monitoring.
- Explainability / feature importance.
- LLM-generated business insights on top of the ML forecast.
- API or UI-based consumption of forecasts.
Repository
GitHub:
https://github.com/genai-dev/manpower-forecasting-product
Technology
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
- Excel / OpenPyXL for input-output processing
Summary
This project converts historical manpower data into a standardized ML forecasting pipeline. Random Forest Regression is used to predict future Closing FLS and Exit values, and business formulas are used to derive Opening FLS, Hiring and Attrition Rate for the next 12 months at Channel × Zone level.
