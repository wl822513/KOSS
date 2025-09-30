# KOSS
Kalman-Optimal Selective State Spaces for Long-Term Sequence Modeling

---

# Project Overview

This project implements **KOSS**, a novel Kalman-Optimal Selective State Space model for long-term sequence modeling.  
KOSS addresses limitations in existing selective SSMs, which often lack theoretical grounding and cannot perform context-aware selection from latent states.  

**Key Contributions:**
- **Kalman-Optimal SSM:** Formulates input-dependent selection as a latent state uncertainty minimization problem using Kalman gain, enabling context-aware, closed-loop selectivity.  
- **Innovation-Driven Selectivity (IDS):** Filters task-irrelevant information while remembering relevant patterns over long horizons.  
- **Spectral Differentiation Unit (SDU):** Performs stable frequency-domain derivative estimation for robust long-sequence modeling.  
- **Segment-wise Parallel Scan:** Hardware-efficient implementation ensuring near-linear scalability, even with dynamic coupling between latent states and parameters.  

**Performance Highlights:**
- Achieves over 79% accuracy on selective copying tasks under high distractor interference.  
- Reduces MSE by 10–30% on nine long-term forecasting benchmarks, outperforming state-of-the-art models in both accuracy and stability.  
- Demonstrates robustness in real-world SSR target tracking under irregular, noisy observations.

---

# Dataset Information

This project uses both **public long-term time series datasets** and a **proprietary dataset** from a real-world Secondary Surveillance Radar (SSR) case study.  

---

## Usage Instructions

1. Download the datasets and place the CSV files in the `data/` folder of the project.  
   - For the SSR dataset, use the folder `data/SSR/`.  
2. Ensure the filenames match those specified in the code.  
3. Due to large file sizes, download options include:
   - Direct download via the provided links  
   - Using Kaggle or GitHub APIs for automated download

---

## 1. Electricity

**Dataset Name**: ElectricityLoadDiagrams20112014  
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014)  
**Description**:  
- Electricity consumption data of 370 clients  
- Time span: 2011-01-01 to 2014-12-31  
- Sampling interval: 15 minutes  
- Format: CSV  
**Example**:


---

## 2. ETT-small

**Dataset Name**: ETT-small (Electricity Transformer Dataset)  
**Source**: [GitHub Repository](https://github.com/zhouhaoyi/ETDataset)  
**Description**:  
- Transformer electricity load data  
- Regions: ETT-small-m1 / ETT-small-m2  
- Time span: 2 years  
- Sampling interval: 1 minute  
- Format: CSV  
**Example**:



---

## 3. Exchange Rate

**Dataset Name**: Exchange Rate Time Series  
**Source**: [GitHub Project](https://github.com/bala-1409/Foreign-Exchange-Rate-Time-Series-Data-science-Project)  
**Description**:  
- Daily exchange rates of multiple currency pairs  
- Time span: varies by currency pair  
- Sampling interval: daily  
- Format: CSV  
**Example**:



---

## 4. Traffic

**Dataset Name**: Traffic Time Series Dataset  
**Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/stealthtechnologies/traffic-time-series-dataset)  
**Description**:  
- Urban road traffic flow data  
- Time span: 1 year  
- Sampling interval: hourly  
- Format: CSV  
**Example**:




---

## 5. Weather

**Dataset Name**: Jena Climate Dataset  
**Source**: [Keras Example](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)  
**Description**:  
- Weather data from Jena, Germany  
- Time span: several years  
- Sampling interval: 10 minutes  
- Features: temperature, humidity, wind speed, pressure, etc. (14 features)  
- Format: CSV  
**Example**:



---

## 6. Secondary Surveillance Radar (SSR) Case Study

**Dataset Name**: SSR Plots for Real-World Trajectory Tracking  
**Source**: [IEEE DataPort](https://ieee-dataport.org/documents/dataset-secondary-surveillance-radar-ssr-plots)  
**DOI**: [10.21227/qxmq-z688](https://doi.org/10.21227/qxmq-z688)  
**Description**:  
- Proprietary dataset created for this study  
- Contains SSR target plots for real-time trajectory prediction and tracking  
- Features: target ID, timestamp, range, azimuth, elevation, velocity, etc.  
- Sampling interval: varies depending on radar acquisition rate  
- Format: CSV  

**Usage Instructions**:  
1. Download the dataset from IEEE DataPort using the DOI link above.  
2. Place the CSV files in the `data/SSR/` folder of the project.  
3. Ensure filenames match those specified in the code.  

**Note**: This dataset validates KOSS’s robustness in handling noisy, sparse, and irregular real-world time series, demonstrating its practical applicability in operational SSR scenarios.


