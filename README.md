# 📊 Employee Training Effectiveness Analyzer

A **Streamlit Dashboard** for visualization of Employees Training Effectiveness, providing trends and predicting Performance Improvement and Promotion Eligibility using Machine Learning Model(MLP Regressor and Random Forest Classifier). 
This project provides insights on the effectiveness of Employee Training on Performance to Promotions leading to more effective data-driven decisions.

---
---

## 📌 Project Summary

As  part of **HCL Tech Internship**, this dashboard was prepared. It allows users to:

- Visualize Employee Training data through Interactive charts
- **Prediction** as per available features


## 🚀 Live App

👉 [Launch the Employee Training Effectiveness Analyzer Webapp](https://employetea.streamlit.app/)


## 💡 Features

- ✅ **Streamlit-powered** user interface
- 📈 Interactive charts using **Plotly** and **Seaborn**
- 🤖 Prediction via **MLP Regressor and Random Forest Classifier**
- 🎛️ Filters by Education level, Work Experience, Training Program and more..


## 🔍 Machine Learning Model

- Trained using the cleaned dataset of ["EXCSV_Cleaned_Data_Wid_Formulas"](https://github.com/Sumi101/A.Employee_TEA/blob/main/EXCSV_Cleaned_Data_Wid_Formulas.csv) placed in the database [ETEA.db](https://github.com/Sumi101/A.Employee_TEA/blob/main/ETEA.db) using SQLite
- Included Features:
  - Education
  - Work Experience
  - Training Program
  - Training Type
  - Pre-Training and Post-Training Scores
  - Peer Learning
  - Feedback Score
- Model Used: `MLP Regressor` & `Random Forest Classifier`
- Model (file): `mlmodel_mlp_rfc.pkl`

### 🔍 More Elements added...
### ⚠️ Note: 
- The current model is a prototype and does not yet deliver prediction for production-level accuracy.
  - Predictions may be inconsistent or imprecise, especially on unseen or edge-case data.
---
---
## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
