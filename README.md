# Kiln Performance Dashboard

**A Streamlit web application for monitoring and analyzing rotary kiln performance metrics in real-time.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-ff4b4b?style=flat-square&logo=streamlit)](https://kilnperformance.streamlit.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME/kiln-performance-dashboard)
[![Python Version](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)](https://python.org)

---

## 📸 Dashboard Preview

### Main Dashboard - Combustion Analysis
*Real-time monitoring of kiln temperatures, LOI, gas consumption, and reactivity*

<img width="1852" height="727" alt="main-dashboard" src="https://github.com/user-attachments/assets/5a6b5a2e-ae1b-4404-84de-dab6a95fc7d9" />
<img width="1852" height="701" alt="temperature-control" src="https://github.com/user-attachments/assets/7eae734e-49e8-43ba-af01-c6bf5761b27e" />

*Figure 1: Main dashboard showing temperature trends, LOI, gas consumption, and reactivity metrics*

### VSD Setpoint vs Kiln Temperature Profile
*Monitor the relationship between kiln speed and temperature zones*

<img width="1846" height="776" alt="gas-consumption" src="https://github.com/user-attachments/assets/dc79cc0e-e4e6-463c-a9da-bb9417324a75" />

*Figure 2: VSD speed vs sintering, burner, and inlet temperatures with target overlays*

### Gas Consumption & Air Flow Analysis
*Track fuel efficiency and airflow optimization*

<img width="1857" height="770" alt="gas-consumption_2" src="https://github.com/user-attachments/assets/71795a2b-58df-4e61-97ca-ba1f391a3420" />

*Figure 3: Gas consumption vs air flow trends with real-time monitoring*

### ML Analytics - LOI Predictor
*AI-powered prediction of Loss on Ignition based on operating conditions*

<img width="1852" height="847" alt="loi-predictor" src="https://github.com/user-attachments/assets/6132b45c-089c-46a0-876b-71e1af993393" />
<img width="1841" height="832" alt="loi-predictor_2" src="https://github.com/user-attachments/assets/2bb21efa-d883-4248-afb4-f411ece94b25" />

*Figure 4: Actual vs predicted LOI with what-if analysis capabilities*

### Process Capability (Cp/Cpk) Analysis
*Statistical process control metrics for quality assurance*

<img width="1842" height="767" alt="process-capability" src="https://github.com/user-attachments/assets/b492b010-ebae-40e2-a62a-f023f0ac5ca8" />
<img width="1852" height="762" alt="process-capability_2" src="https://github.com/user-attachments/assets/1df1a04e-93c8-4f29-893f-ee4a43b117fd" />

*Figure 5: Cp/Cpk analysis with histogram and specification limits*

### Control Charts (X-bar & R / I-MR)
*Statistical process control charts for monitoring process stability*

<img width="1842" height="802" alt="control-charts" src="https://github.com/user-attachments/assets/026263db-3401-4d4d-86f5-984811f4e4f1" />
<img width="1852" height="652" alt="image" src="https://github.com/user-attachments/assets/78c9a839-666f-43af-9f32-a38b84dfc7b9" />

*Figure 6: X-bar and R control charts with out-of-control detection*


---

## 📊 Overview

The **Kiln Performance Dashboard** is a comprehensive data visualization and analytics tool built for the cement manufacturing industry. It addresses a critical challenge: while cement plants generate vast amounts of process data from DCS, SCADA, LIMS, and weighbridges, this data often exists in disconnected systems.

This dashboard bridges that gap by providing **real-time visibility** into key kiln performance indicators, enabling operators and engineers to:

- Monitor burning zone temperature and kiln shell temperature profile
- Track specific heat consumption (kcal per kg clinker)
- Analyze free lime and clinker-to-cement ratio trends
- View Overall Equipment Effectiveness (OEE) for kiln operations
- Identify deviations and take corrective action within the same shift

---

## ✨ Features

### 🔥 Real-Time Kiln Monitoring
- **Live temperature tracking** for burning zone and kiln shell surface
- **Temperature profile visualization** across the kiln length (sintering, burner, inlet)
- **VSD speed monitoring** with temperature correlation
- **Alarm triggers** when parameters drift outside tolerance ranges

### 📈 Key Performance Indicators (KPIs)
- **Loss on Ignition (LOI)** – product quality metric
- **Reactivity (seconds)** – product quality indicator
- **Gas Consumption (m³)** – fuel efficiency metric
- **Air Flow (%)** – combustion optimization

### 🤖 Machine Learning Analytics
- **LOI Prediction Model** – Random Forest regression to predict LOI from operating conditions
- **Gas Optimization Model** – Predict and optimize gas consumption
- **Quality Classification Model** – Predict if quality targets will be met (LOI < 5%, Reactivity < 90 sec)
- **Anomaly Detection** – Isolation Forest to identify unusual operating conditions

### 📊 Process Capability Analysis
- **Cp, Cpk, Cpu, Cpl** calculations
- **Process capability histograms** with specification limits
- **Defect rate estimation** and Sigma Level calculation

### 📉 Statistical Process Control
- **X-bar & R Charts** – monitor process mean and variability
- **I-MR Charts** – individual and moving range control charts
- **Out-of-control detection** with visual alerts

### 🎛️ User-Friendly Interface
- **Tlowana Resources branding** with custom color theme
- **Role-based views** – operators, maintenance engineers, and plant heads see tailored dashboards
- **Filterable data** by date range, process stage, and parameter
- **Responsive design** accessible on PC monitors in CCR rooms or on mobile devices

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-learn (Random Forest, Isolation Forest) |
| **Visualization** | Plotly, Plotly Express |
| **Statistical Analysis** | Scipy.stats for capability analysis |
| **Data Sources** | Excel/CSV upload via Pandas |
| **UI Enhancements** | Custom CSS with Company branding |

---
