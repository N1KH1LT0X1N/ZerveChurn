# User Behavior Analytics: Comprehensive Data Analysis Project

**Author:** Data Science Team  
**Last Updated:** March 2026

---

## 📋 Project Overview

This project provides an end-to-end analysis of user retention and behavior patterns using event-level data. The analysis encompasses data quality assessment, exploratory analysis, segmentation, feature engineering, predictive modeling, and advanced analytics techniques to derive actionable insights for user engagement and retention strategies.

---

## 🗺️ Navigation Guide

This canvas contains **49 blocks** organized into logical analytical workflows:

### Main Analysis Flow (Left to Right):
1. **Data Foundation** → Example Dataset → Data Exploration
2. **Statistical Profiling** → Statistical summaries and data quality checks
3. **Segmentation** → Engagement, workflow pattern, temporal & monetization segments
4. **Feature Engineering** → Event taxonomy, session patterns, lifecycle stages
5. **Predictive Modeling** → Success metrics, ensemble models, survival analysis
6. **Advanced Analytics** → Anomaly detection, collaboration networks, churn prediction
7. **Insights & Reporting** → Dashboard synthesis and comprehensive findings

---

## 🚀 How to Run This Project

### Prerequisites:
- All data is pre-loaded (`user_retention.parquet`)
- Required libraries: pandas, numpy, matplotlib, scikit-learn, lifelines

### Execution Order:
1. Start with **Example Dataset** block to load data
2. Run **Data Exploration** to understand data structure
3. Execute blocks sequentially following the DAG connections
4. Key analysis blocks will auto-run and generate visualizations
5. Review final insights in **Integrated Dashboard Synthesis** and **Advanced Analysis Synthesis**

---

## 📊 Section Overview

### 1. **Data Exploration**
   - **Blocks:** Example Dataset, Data Exploration, Statistical Summaries, Data Quality Checks, Temporal Coverage
   - **Key Outputs:** Data quality metrics, schema overview, missing value analysis
   - **Purpose:** Understand data structure, quality, and coverage

### 2. **Segmentation**
   - **Blocks:** Engagement Segmentation, Workflow Pattern Segmentation, Temporal & Monetization Segmentation
   - **Key Outputs:** K-means clusters, workflow patterns, temporal segments
   - **Purpose:** Identify distinct user groups for targeted strategies

### 3. **Feature Engineering**
   - **Blocks:** Event Taxonomy & Categorization, Workflow Stage Mapping, Session Pattern Analysis, Comprehensive Feature Engineering
   - **Key Outputs:** 141 events → 22 categories → 14 workflow stages; 29 engineered features
   - **Purpose:** Transform raw event data into meaningful features for modeling

### 4. **Modeling**
   - **Blocks:** Primary Success Metrics, Composite Success Score & Labeling, Validation & Business Alignment, Data Prep & Train/Val/Test Split, Base Models & Ensemble
   - **Key Outputs:** Random Forest (99.88% test accuracy), ensemble models saved to `ensemble_models.pkl`
   - **Purpose:** Predict user success and identify key drivers

### 5. **Advanced Analytics**
   - **Blocks:** Survival Analysis Data Preparation, Kaplan-Meier Survival Curves, Churn Risk Scoring, Isolation Forest Anomaly Detection, Collaboration Network & Centrality, Community Detection & Success Correlation
   - **Key Outputs:** Survival curves, churn probability scores, 14 high-risk alerts, 271 anomalous users
   - **Purpose:** Advanced predictive insights and behavioral pattern detection

### 6. **Synthesis & Insights**
   - **Blocks:** Advanced Analysis Synthesis, Integrated Dashboard Synthesis, Comprehensive User Analysis Findings
   - **Key Outputs:** Unified user profiles, actionable segment insights, 5 priority action groups
   - **Purpose:** Translate technical findings into business actions

### 7. **Visualization & Reporting**
   - **Blocks:** Interactive Visualizations & Segment Export, Hierarchical Event Visualization, Engagement Momentum Tracking, Growth Trajectory Classification
   - **Key Outputs:** 20+ professional charts using Zerve design system, master_segments DataFrame
   - **Purpose:** Visual exploration and presentation-ready charts

---

## ✅ Key Deliverables

- [x] **Data Quality Report** — 409,287 events, 5,410 users, 96 sparse columns analyzed
- [x] **Event Taxonomy** — 141 events → 22 categories → 14 workflow stages
- [x] **User Segmentation** — 5 engagement clusters, lifecycle stages, temporal patterns
- [x] **Session Analysis** — 12,641 sessions with deep work detection
- [x] **Success Metrics Framework** — Composite scoring with business validation
- [x] **Predictive Models** — 6 ensemble models (99.88% test accuracy)
- [x] **Survival Analysis** — Kaplan-Meier curves by risk segment
- [x] **Churn Prediction** — 3,216 active users scored; 14 high-risk alerts
- [x] **Anomaly Detection** — 271 exceptional users via Isolation Forest
- [x] **Network Analysis** — Collaboration graph with centrality & community detection
- [x] **Visualizations** — 20+ professional Zerve-styled charts
- [x] **Reports** — Executive summary + comprehensive markdown reports

---

## 🔑 Key Findings

- **5,410 unique users** across **409,287 events** over 98 days
- **Power law distribution**: top 5% (273 users) average 544 events vs. ~5 for bottom 62%
- **99.88% model accuracy** in predicting user success (Random Forest)
- **14 high-risk users** flagged for immediate retention intervention
- **31–60 day period** identified as peak churn vulnerability window (43% of churn)
- **271 anomalous users** identified with unusual behavioral patterns
- **12,641 sessions** analyzed; 28% qualify as "deep work" sessions

---

## 📧 Contact & Support

For questions or additional analysis requests, contact the Data Science Team.

**Project Status:** ✅ Complete — All deliverables ready