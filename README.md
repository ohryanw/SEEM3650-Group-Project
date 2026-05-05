# Predicting Monthly PM2.5 Concentrations in Hong Kong with Weather and Vehicle Factors

## Project Title

**Report on Predicting Monthly PM2.5 Concentrations in Hong Kong with Weather and Vehicle Factors**

## Group Members

- Mak Loren Tsz Long 1155212355
- Leung Ting Yan Windsor 1155213897
- Woo Ryan 1155214118

## GitHub Repository

This repository contains the Python code, data processing workflow, exploratory data analysis, machine learning models, visualizations, and final report for predicting monthly PM2.5 concentrations in Hong Kong using weather and vehicle-related variables.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Question](#research-question)
3. [Dataset Description](#dataset-description)
4. [Target Variable](#target-variable)
5. [Predictor Variables](#predictor-variables)
6. [Data Sources](#data-sources)
7. [Station Weighting Method](#station-weighting-method)
8. [Methodology](#methodology)
9. [Data Processing Workflow](#data-processing-workflow)
10. [Model Design](#model-design)
11. [Feature Sets](#feature-sets)
12. [Train-Test Strategy](#train-test-strategy)
13. [Evaluation Metrics](#evaluation-metrics)
14. [Exploratory Data Analysis](#exploratory-data-analysis)
15. [Model Results](#model-results)
16. [Actual vs Predicted Plots](#actual-vs-predicted-plots)
17. [Best Model Discussion](#best-model-discussion)
18. [Feature Importance](#feature-importance)
19. [Residual Analysis](#residual-analysis)
20. [Pros and Cons](#pros-and-cons)
21. [Potential Improvements](#potential-improvements)
22. [Societal Impact](#societal-impact)
23. [Conclusion](#conclusion)
24. [How to Run the Code](#how-to-run-the-code)
25. [Repository Structure](#repository-structure)
26. [Requirements](#requirements)
27. [Notes About Figure Paths](#notes-about-figure-paths)
28. [Acknowledgements](#acknowledgements)
29. [Disclaimer](#disclaimer)

---

## Project Overview

This project investigates whether monthly PM2.5 concentrations in Hong Kong can be predicted using meteorological conditions and vehicle-related factors.

PM2.5 refers to fine particulate matter with a diameter of 2.5 micrometers or smaller. It is an important air pollutant because it can affect public health, visibility, and environmental quality.

The project combines monthly data from 2015 to 2025, including:

1. PM2.5 air pollution data
2. Weather data
3. Vehicle registration data

The purpose of the project is not only to build a predictive model, but also to examine whether adding vehicle-related variables improves prediction accuracy beyond using weather variables alone.

In particular, this project studies whether changes in Hong Kong's vehicle fleet, including the growth of electric vehicles and changes in diesel vehicle share, are associated with changes in monthly PM2.5 levels.

---

## Research Question

The main research question is:

> Can monthly PM2.5 concentrations in Hong Kong be predicted using weather and vehicle-related variables, and does adding vehicle information improve prediction compared with weather-only models?

The project also considers the following questions:

- How strongly are PM2.5 levels related to weather conditions?
- Do vehicle-related variables improve model performance?
- Which machine learning model gives the best prediction accuracy?
- Which features are most important in predicting PM2.5?
- Is the rise of electric vehicles associated with lower PM2.5 levels?

---

## Dataset Description

The dataset covers monthly observations from:

```text
January 2015 to December 2025
