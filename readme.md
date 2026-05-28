# CricAI – Real-Time Cricket Win Probability Predictor

## 🏏 About the Project

CricAI is an end-to-end Machine Learning project that predicts the winning probability of a cricket team during a live T20 match.

The system analyzes:
- Current match situation
- Ball-by-ball match states
- Team performance
- Match pressure
- Historical cricket data

and predicts the chances of winning for both teams in real time.

This project supports:
- IPL Teams
- International Teams
- Multiple Venues
- Custom Match Scenarios

---

# 🚀 Features

✅ Real-time win probability prediction  
✅ IPL + International teams support  
✅ Interactive sports analytics dashboard  
✅ Ball-by-ball match state analysis  
✅ Machine Learning based predictions  
✅ Modern dark-themed UI  
✅ End-to-end ML pipeline  
✅ Calibrated probability predictions  

---

# 🧠 Machine Learning Workflow

```text
Raw JSON Match Data
        ↓
Data Building
        ↓
Feature Engineering
        ↓
Data Transformation
        ↓
Model Training
        ↓
Probability Calibration
        ↓
Prediction Pipeline
        ↓
Interactive Dashboard
```

---

# 📊 Model Information

The project uses:

- XGBoost Classifier
- Platt Probability Calibration
- Ball-by-ball feature engineering
- Match-wise temporal data splitting

### Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.9179 |
| Accuracy | 83% |
| Training Rows | 387K+ |
| Matches Trained | 6,883 |

---

# 🛠️ Technologies Used

## Frontend
- Streamlit
- HTML
- CSS
- Plotly

## Backend
- Python

## Machine Learning
- XGBoost
- Scikit-learn
- Pandas
- NumPy

## Deployment
- Streamlit Cloud
- GitHub

---

# 📂 Project Structure

```text
cricai_project/

│
├── artifacts/
│   ├── calibrated_model.pkl
│   ├── preprocessor.pkl
│   └── feature_columns.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── logger.py
│   ├── utils.py
│   └── exception.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
```

---

## 2️⃣ Move into Project Folder

```bash
cd cricai_project
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv cricket_ml
```

### Activate Environment

```bash
cricket_ml\Scripts\activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Start Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# 📈 How Prediction Works

The model takes:
- Batting Team
- Bowling Team
- Current Score
- Target
- Overs Completed
- Wickets
- Venue

Then predicts:
- Batting Team Win %
- Bowling Team Win %

using trained historical match data.

---

# 🌐 Deployment

The project is deployed using:
- GitHub
- Streamlit Community Cloud

---

# 🎯 Future Improvements

- ODI & Test match support
- Live API integration
- Player-based prediction
- Advanced analytics dashboard
- Match momentum tracking
- Dynamic team rankings

---

# 👨‍💻 Author

**Aravind Madishetty**

Machine Learning & AI Enthusiast  
Focused on building real-world AI and analytics systems.

---

# ⭐ Project Highlights

✅ End-to-End ML Project  
✅ Real-Time Prediction System  
✅ Sports Analytics Application  
✅ Production-Style ML Pipeline  
✅ Interactive Dashboard UI  
✅ Probability Calibration Techniques
