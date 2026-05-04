# Predicting Monthly PM2.5 Concentrations in Hong Kong Using Weather and Vehicle Factors

## Project Overview

This project predicts monthly PM2.5 concentrations in Hong Kong using weather conditions and vehicle-related variables. PM2.5 is a fine particulate air pollutant that can negatively affect public health, so understanding its patterns is important for environmental planning and transport policy.

The study combines monthly PM2.5 data, weather data, and vehicle registration data from 2015 to 2025. Machine learning regression models are then used to estimate monthly PM2.5 levels and compare whether adding vehicle-related variables improves prediction accuracy compared with using weather variables alone.

The project also investigates whether the increasing share of electric vehicles is associated with changes in PM2.5 concentrations over time.

---

## Research Question

The main research question is:

> Can monthly PM2.5 levels in Hong Kong be predicted using weather and vehicle data, and does adding vehicle-related information improve prediction accuracy compared with weather-only models?

More specifically, the project compares:

1. Weather-only models
2. Weather + vehicle models
3. Weather + vehicle + diesel-share models

---

## Group Members

- Mak Loren Tsz Long  
- Leung Ting Yan Windsor  
- Woo Ryan  

---

## Repository Contents

This repository contains the code, figures, and report materials for the project.

Suggested repository structure:

```text
SEEM3650-Group-Project/
│
├── README.md
├── report/
│   └── PM25_prediction_report.pdf
│
├── data/
│   ├── pm25_data.csv
│   ├── weather_data.csv
│   ├── vehicle_data.csv
│   └── cleaned_monthly_dataset.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   └── model_pipeline.py
│
├── figures/
│   ├── weighted_pm25_trend.png
│   ├── weather_trends.png
│   ├── vehicle_trends.png
│   ├── histograms.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted_linear_weather_only.png
│   ├── actual_vs_predicted_ridge_weather_only.png
│   ├── actual_vs_predicted_randomforest_weather_only.png
│   ├── actual_vs_predicted_linear_weather_vehicles.png
│   ├── actual_vs_predicted_ridge_weather_vehicles.png
│   ├── actual_vs_predicted_randomforest_weather_vehicles.png
│   ├── actual_vs_predicted_linear_weather_vehicles_diesel.png
│   ├── actual_vs_predicted_ridge_weather_vehicles_diesel.png
│   ├── actual_vs_predicted_randomforest_weather_vehicles_diesel.png
│   ├── feature_importance.png
│   └── residual_plot.png
│
└── requirements.txt
