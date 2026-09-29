# 🛡️ IEEE-CIS Fraud Detection

> An end-to-end machine learning project for detecting potentially fraudulent online transactions using **CatBoost, LightGBM, XGBoost, ensemble learning, and stacking**, with a **FastAPI backend and Streamlit frontend**.

The project covers the complete ML workflow, from data exploration and feature engineering to cross-validation, model evaluation, OOF ensembling, stacking, and local application development.

<p align="center">
  ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
  ![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow)
  ![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-green)
  ![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
  ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
  ![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit)
</p>

---

## 📌 Project Overview

The objective of this project is to predict the target variable:

**`isFraud`**

| Target | Meaning                    |
| ------ | -------------------------- |
| `0`    | Non-fraudulent transaction |
| `1`    | Fraudulent transaction     |

The IEEE-CIS dataset contains transaction and identity-related information, with hundreds of anonymized features and substantial missing values. Fraud detection is an imbalanced classification problem, so model evaluation focuses on metrics such as ROC-AUC and PR-AUC in addition to threshold-based classification results.

## 🌐 Live Demo

Try the deployed application here:

🔗 **[IEEE-CIS Fraud Detection — Live App](https://ieee-cis-fraud-detection-kr9phkxkggjugdkqzhcy76.streamlit.app/)**

The application allows users to upload transaction data in CSV format, generate fraud probabilities, view predicted fraud cases and download the prediction results.




| Project detail        | Description                               |
| --------------------- | ----------------------------------------- |
| 🎯 Problem type       | Binary classification                     |
| 📦 Dataset            | IEEE-CIS Fraud Detection                  |
| 📊 Training records   | 590,540                                   |
| 🧩 Data sources       | Transaction and identity tables           |
| 🏷️ Target             | `isFraud`                                 |
| ⚖️ Data challenge     | Class imbalance and missing values        |
| 🤖 Base models        | CatBoost, LightGBM, XGBoost               |
| 🧠 Final architecture | Stacking with Random Forest               |
| 🌐 Application        | FastAPI + Streamlit                       |
| 🚦 Deployment status  | Tested locally with available public deployment |

## 🎯 Project Objectives

- 🧹 Explore and clean a large real-world dataset.
- 🔎 Analyze missing values and feature distributions.
- 🛠️ Perform feature engineering and preprocessing.
- 🤖 Train multiple gradient-boosting models.
- 🔁 Implement 5-fold cross-validation and OOF predictions.
- 🎲 Experiment with multiple random seeds.
- ⚖️ Compare equal-weight and weighted ensembles.
- 🧠 Implement stacking with a meta-model.
- 📈 Evaluate using ROC-AUC and PR-AUC.
- 🔌 Build a reusable model inference pipeline.
- 🚀 Create a REST API and interactive prediction interface.

## 📦 Dataset

**Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)

The dataset is provided as separate transaction and identity files.

| Dataset                   | Description                          |
| ------------------------- | ------------------------------------ |
| Transaction               | Transaction-level features           |
| Identity                  | Device and identity-related features |
| Join key                  | `TransactionID`                      |
| Target                    | `isFraud`                            |
| Training transaction rows | 590,540                              |

The transaction and identity datasets are merged using `TransactionID`. The resulting data contains numerical and categorical features, including anonymized variables and columns with substantial missingness.

> ⚠️ The raw dataset is not included in this repository. It must be downloaded from Kaggle. The `data/` directory is excluded from Git.

## 🗂️ Project Structure

```text
## 📁 Project Structure

IEEE-CIS-Fraud-Detection/
│
├──📁app/
│   ├── feature_engineering.py
│   ├── test_features.py
│   ├── inference.py
│   ├── main.py
│   ├── test_api.py
│   ├── test_batch_api.py
│   └── streamlit_app.py
│
├──📁data/                          # Excluded from Git
│   ├── train_transaction.txt
│   ├── train_identity.txt
│   ├── test_transaction.txt
│   ├── test_identity.csv
│   ├── train_engineered.pkl
│   ├── test_engineered.pkl
│   └── training_stats.pkl
│
├──📁models/
│   ├── catboost_final.cbm
│   ├── catboost_feature_columns.pkl
│   ├── catboost_categorical_columns.pkl
│   ├── xgb_seed_123.json
│   ├── xgb_seed_2024.json
│   ├── xgb_seed_777.json
│   ├── xgb_preprocessor.pkl
│   ├── lgb_seed_123.txt
│   ├── lgb_seed_2024.txt
│   ├── lgb_seed_777.txt
│   ├── lgb_preprocessor.pkl
│   ├── final_meta_model.pkl
│   └── meta_feature_columns.pkl
│
├──📁notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_baseline_models.ipynb
│   ├── 4.1_CatBoost_5Fold_CV.ipynb
│   ├── 4.2_XGBoost_5Fold_CV.ipynb
│   ├── 4.3_LightGBM_5Fold_CV.ipynb
│   ├── 5_Baseline_Ensembling.ipynb
│   ├── 6.1_XGBoost_Seed_Experiment.ipynb
│   ├── 6.2_LightGBM_Seed_Experiment.ipynb
│   ├── 7_Weighted_Ensembling.ipynb
│   ├── 8_Stacking.ipynb
│   ├── 9_Catboost_Test_Predictions.ipynb
│   ├── 10_XGBoost_Test_Predictions.ipynb
│   ├── 11_LightGBM_Test_Predictions.ipynb
│   ├── 12_Final_Model_and_Submission.ipynb
│   └── 13_Kaggle_Leaderboard_analysis.ipynb
│
├──📁outputs/
│   ├──📁oof/
│   │   ├── catboost_oof.csv
│   │   ├── lightgbm_oof.csv
│   │   ├── lightgbm_seed_123_oof.csv
│   │   ├── lightgbm_seed_2024_oof.csv
│   │   ├── lightgbm_seed_777_oof.csv
│   │   ├── stacking_meta_oof.csv
│   │   ├── xgboost_oof.csv
│   │   ├── xgboost_seed_123_oof.csv
│   │   ├── xgboost_seed_2024_oof.csv
│   │   └── xgboost_seed_777_oof.csv
│   │
│   └──📁predictions/
│       ├── catboost_test_predictions.csv
│       ├── lightgbm_test_predictions.csv
│       └── xgboost_test_predictions.csv
│
├── src/
│
├──📄.gitignore
├──📄README.md
├──📄requirements.txt
└──📄submission.csv
```

_The tree shows the main project organization; generated files may vary as experiments evolve._

## 🔄 Machine Learning Workflow

| Stage                       | Notebook / component                   | Purpose                                                           |
| --------------------------- | -------------------------------------- | ----------------------------------------------------------------- |
| 1️⃣ Data understanding & EDA | `01_data_understanding_eda.ipynb`      | Inspect data, target distribution, missingness, and feature types |
| 2️⃣ Feature engineering      | `02_feature_engineering.ipynb`         | Merge data, extract features, and prepare model inputs            |
| 3️⃣ Baseline models          | `03_baseline_models.ipynb`             | Train initial CatBoost, LightGBM, and XGBoost models              |
| 4️⃣ 5-fold CV                | CV notebooks                           | Generate OOF predictions and evaluate model performance           |
| 5️⃣ Baseline ensembling      | `05_baseline_ensembling.ipynb`         | Combine base-model predictions                                    |
| 6️⃣ Seed experiments         | `06_seed_experiments.ipynb`            | Compare predictions from multiple random seeds                    |
| 7️⃣ Weighted ensembling      | `07_weighted_ensembling.ipynb`         | Experiment with model-specific weights                            |
| 8️⃣ Stacking                 | `08_stacking.ipynb`                    | Train a meta-model on base-model prediction features              |
| 9️⃣ Final predictions        | `09_Final_Model_and_Predictions.ipynb` | Generate final test predictions and submission                    |
| 🔟 Application              | `app/`                                 | Run inference through FastAPI and Streamlit                       |

## 🧹 Data Preprocessing & Feature Engineering

The original merged training data contained **590,540 rows and 394 columns**. Feature engineering expanded the feature set, and the final inference pipeline aligns data to a schema of **476 columns**.

| Area                    | Approach                                                         |
| ----------------------- | ---------------------------------------------------------------- |
| 🔗 Data merging         | Transaction and identity tables joined using `TransactionID`     |
| 🕒 Time features        | Derived features from `TransactionDT`                            |
| 🧩 Categorical features | Prepared for model-specific handling                             |
| 🧮 Numerical features   | Processed using saved preprocessing artifacts                    |
| 🕳️ Missing values       | Investigated and handled according to feature/model requirements |
| 📐 Feature alignment    | Consistent 476-column inference schema                           |
| ♻️ Training statistics  | Saved and reused during inference                                |

Training-derived statistics are stored in `training_stats.pkl` and reused by the inference pipeline to maintain consistency between training and prediction.

The main feature engineering and inference modules are:

- `app/feature_engineering.py`
- `app/inference.py`

## 🤖 Models Evaluated

Three gradient-boosting algorithms were used as base models.

| Model            | Role         | Key characteristics                         |
| ---------------- | ------------ | ------------------------------------------- |
| 🐈 CatBoost      | Base learner | Handles categorical features                |
| ⚡ LightGBM      | Base learner | Gradient boosting with multiple seed models |
| 🌳 XGBoost       | Base learner | Gradient boosting with multiple seed models |
| 🌲 Random Forest | Meta-model   | Learns from base-model predictions          |

### ⚙️ Selected Hyperparameters

**🐈 CatBoost**

| Parameter       | Value |
| --------------- | ----: |
| `iterations`    |   300 |
| `learning_rate` |  0.05 |
| `depth`         |     6 |
| `random_seed`   |    42 |

**⚡ LightGBM**

| Parameter           |                Value |
| ------------------- | -------------------: |
| `n_estimators`      |                  100 |
| `learning_rate`     |                 0.05 |
| `num_leaves`        |                   31 |
| `max_depth`         |                   -1 |
| `min_child_weight`  |                    5 |
| `min_child_samples` |                   20 |
| `subsample`         |                  0.8 |
| `subsample_freq`    |                    1 |
| `colsample_bytree`  |                  0.8 |
| Seeds               | `123`, `2024`, `777` |

**🌳 XGBoost**

| Parameter          |                Value |
| ------------------ | -------------------: |
| `n_estimators`     |                  100 |
| `max_depth`        |                   10 |
| `subsample`        |                  0.8 |
| `colsample_bytree` |                  0.8 |
| Seeds              | `123`, `2024`, `777` |

**🌲 Random Forest Meta-Model**

| Parameter      | Value |
| -------------- | ----: |
| `n_estimators` |   200 |
| `max_depth`    |     5 |
| `random_state` |    42 |

## 📊 Model Evaluation

ROC-AUC and PR-AUC were used to evaluate model performance, with particular attention to the imbalanced target distribution.

### 🔬 5-Fold OOF Performance

| Model                     |    ROC-AUC |     PR-AUC |
| ------------------------- | ---------: | ---------: |
| 🐈 CatBoost               |     0.9121 |     0.5943 |
| ⚡ LightGBM               |     0.8944 |     0.5470 |
| 🌳 XGBoost                |     0.8824 |     0.5224 |
| 🤝 Equal-weight ensemble  |     0.9208 |     0.6127 |
| 🧠 Random Forest stacking | **0.9266** | **0.6330** |

_These are the recorded local OOF evaluation results._

### ⚖️ Weighted Ensembling

A manually tested weighted ensemble combined the base-model predictions as follows:

| Model       |   Weight |
| ----------- | -------: |
| 🐈 CatBoost |     0.50 |
| ⚡ LightGBM |     0.25 |
| 🌳 XGBoost  |     0.25 |
| **Total**   | **1.00** |

The weighted ensemble achieved a recorded validation ROC-AUC of **0.923281**.

### 🧠 Stacking

Unlike simple averaging, stacking uses a separate model to learn how base-model predictions can be combined.

In this project, out-of-fold predictions were used as meta-features to train a Random Forest meta-model, helping avoid training the meta-model directly on in-sample base predictions.

| Ensemble strategy      | Description                                     |
| ---------------------- | ----------------------------------------------- |
| Equal-weight averaging | Average base-model predictions                  |
| Seed-based averaging   | Average predictions from different seeds        |
| Weighted averaging     | Combine base models using selected weights      |
| Stacking               | Train a meta-model using base-model predictions |

## 🏗️ Final Stacking Architecture

The final inference architecture uses one CatBoost model, three XGBoost seed models, three LightGBM seed models, and a Random Forest meta-model.

| Component        | Configuration                 | Output                  |
| ---------------- | ----------------------------- | ----------------------- |
| 🐈 CatBoost      | One final model               | Fraud probability       |
| 🌳 XGBoost       | 3 seeds: `123`, `2024`, `777` | Averaged probability    |
| ⚡ LightGBM      | 3 seeds: `123`, `2024`, `777` | Averaged probability    |
| 🧩 Meta-features | Base-model outputs            | Input to meta-model     |
| 🌲 Random Forest | 200 estimators, depth 5       | Final fraud probability |

```text
                📥 Transaction Data
                        │
                        ▼
               🧹 Feature Engineering
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
      🐈 CatBoost     🌳 XGBoost    ⚡ LightGBM
          │          3 seed models  3 seed models
          │             │             │
          │          Averaging     Averaging
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                  🧩 Meta-Features
                        │
                        ▼
                 🌲 Random Forest
                   Meta-Model
                        │
                        ▼
                🎯 Fraud Probability
```

## 🏆 Kaggle Results

The final stacking architecture was used to generate the Kaggle submission.

| Leaderboard            |  ROC-AUC |
| ---------------------- | -------: |
| 🌐 Public leaderboard  | 0.942408 |
| 🔒 Private leaderboard | 0.921572 |

These are the recorded competition leaderboard scores and are distinct from the project's local OOF metrics.

## 🌐 Prediction Application

The project includes a locally tested application built with **FastAPI** and **Streamlit**.

### 🔌 FastAPI Backend

FastAPI serves the saved model artifacts and returns fraud probabilities through REST endpoints.

| Method | Endpoint         | Description                   |
| ------ | ---------------- | ----------------------------- |
| `GET`  | `/`              | Root endpoint                 |
| `GET`  | `/health`        | Health check                  |
| `POST` | `/predict`       | Single-transaction prediction |
| `POST` | `/predict_batch` | Batch prediction              |

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### 🖥️ Streamlit Frontend

The frontend allows users to upload transaction CSV files, inspect the data, generate predictions, and download the results.

| Feature             | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| 📤 CSV upload       | Upload transaction records                                         |
| 👀 Data preview     | View the uploaded data                                             |
| 🚀 Batch prediction | Send multiple transactions to FastAPI                              |
| 📊 Summary metrics  | View total records, predicted fraud count, and average probability |
| 🧾 Results table    | Inspect predictions for each transaction                           |
| 📥 Download         | Export predictions to CSV                                          |

The prediction output includes:

| Column              | Description                            |
| ------------------- | -------------------------------------- |
| `fraud_probability` | Estimated probability of fraud         |
| `predicted_fraud`   | Class prediction using a 0.5 threshold |

The probability is a model estimate. The binary class is determined by comparing that probability with the application's current threshold of 0.5.

### 🧪 Local Testing

| Test                    | Result                 |
| ----------------------- | ---------------------- |
| Single-transaction API  | Successful             |
| 5-row batch API         | Successful             |
| 5,000-row CSV upload    | Processed successfully |
| Prediction CSV download | Available              |
| Public cloud deployment | Not yet completed      |

In a 5,000-row test, the interface reported **81 predicted fraud transactions** and an **average fraud probability of 3.15%**. These are model predictions, not verified fraud labels.


## 🛠️ Technologies Used

| Technology              | Purpose                                 |
| ----------------------- | --------------------------------------- |
| 🐍 Python               | Core development                        |
| 🐼 Pandas               | Data manipulation                       |
| 🔢 NumPy                | Numerical computation                   |
| 📊 Matplotlib / Seaborn | Data visualization                      |
| 🧰 Scikit-learn         | Preprocessing, metrics, and meta-model  |
| 🐈 CatBoost             | Gradient boosting                       |
| ⚡ LightGBM             | Gradient boosting                       |
| 🌳 XGBoost              | Gradient boosting                       |
| 📓 Jupyter Notebook     | Experiments                             |
| 🚀 FastAPI              | REST API backend                        |
| 🦄 Uvicorn              | ASGI server                             |
| 🖥️ Streamlit            | Interactive frontend                    |
| 🔗 Requests             | HTTP communication                      |
| 🔀 Git / Git LFS        | Version control and large-file handling |

## ⚙️ Installation

### 1. 📥 Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "IEEE CIS Fraud Detection"
```

### 2. 🐍 Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🗃️ Prepare Model Artifacts

The inference application requires the saved model and preprocessing artifacts used by `app/inference.py`, including:

- CatBoost model and feature metadata
- XGBoost seed models and preprocessor
- LightGBM seed models and preprocessor
- Random Forest meta-model and meta-feature metadata
- Training-derived feature statistics

The raw Kaggle dataset is not required for ordinary inference if all required model and preprocessing artifacts are available.

> ⚠️ The CatBoost model is approximately 210.62 MB, which exceeds GitHub's 100 MB regular Git file limit. It must be stored with Git LFS or another model-artifact storage solution. If the repository uses Git LFS, install it and run `git lfs install` and `git lfs pull` after cloning.

## ▶️ Running the Application Locally

Run the backend and frontend in **separate terminals** from the project root.

### 🚀 Terminal 1 — FastAPI Backend

```powershell
python -m uvicorn main:app --app-dir app --reload
```

Backend:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

### 🖥️ Terminal 2 — Streamlit Frontend

```powershell
python -m streamlit run app/streamlit_app.py
```

Frontend:

`http://localhost:8501`

Both processes must remain running while using the local application. Stop a service with `Ctrl + C` in its terminal.

## 🚀 Deployment

The Streamlit application is deployed using **Streamlit Community Cloud** and is publicly accessible through a live URL.

* **Frontend:** Streamlit
* **Inference:** Integrated Python ML pipeline
* **Models:** CatBoost, XGBoost, LightGBM and Random Forest stacking model
* **Hosting:** Streamlit Community Cloud

The deployed application loads the trained models and preprocessors and performs feature engineering and inference directly when a user uploads a CSV file.

FastAPI is also included in the project for API-based inference and local testing, but the currently deployed Streamlit application uses direct inference.

## 📥 Input Format

The Streamlit application accepts CSV files containing transaction-level features compatible with the inference pipeline.

| Requirement                      | Details                                                  |
| -------------------------------- | -------------------------------------------------------- |
| File format                      | CSV                                                      |
| Data                             | Transaction-level records                                |
| Feature schema                   | Must be compatible with the feature engineering pipeline |
| Output                           | Original input columns plus prediction columns           |
| Current classification threshold | 0.5                                                      |

The pipeline performs required feature transformations and aligns the data to the expected model schema. Arbitrary CSV files are not guaranteed to be compatible.

## 🖥️ How to Use

1. Open the [Live Application](https://ieee-cis-fraud-detection-kr9phkxkggjugdkqzhcy76.streamlit.app/).
2. Upload a CSV file containing raw transaction data with the expected input features.
3. Click **Predict Fraud**.
4. View the fraud probabilities and predicted fraud labels.
5. Download the results as a CSV file.

The current classification threshold is 0.5. Predicted labels are based on this threshold, while fraud probabilities are also provided for further analysis.

**Note:** This application is intended for demonstration and educational purposes. Do not upload sensitive financial or personal information.

## ☁️ Deployment Test

The deployed application was successfully tested with a batch of 10,000 transactions.

| Metric                    | Result |
| ------------------------- | -----: |
| Transactions processed    | 10,000 |
| Predicted fraud cases     |    151 |
| Predicted fraud rate      |  1.51% |
| Average fraud probability |  3.22% |

These are application inference test results, not ground-truth model accuracy metrics.


## 💡 Key Learnings

- 🧹 Data cleaning and missing-value analysis on large datasets.
- 🧩 Merging transaction and identity information.
- 🛠️ Consistent feature engineering for training and inference.
- ⚖️ Model evaluation on imbalanced classification data.
- 🔁 Cross-validation and out-of-fold predictions.
- 🎲 Seed-based model experimentation.
- 🤝 Equal and weighted ensembling.
- 🧠 Stacking and meta-model training.
- 🔌 Building an inference API with FastAPI.
- 🖥️ Creating an interactive prediction interface with Streamlit.
- 📦 Managing saved model artifacts for application use.

A key part of the project was comparing validation performance using OOF predictions and exploring how multiple models can complement one another.

## 🚧 Limitations & Future Improvements

| Area             | Potential improvement                                                                  |
| ---------------- | -------------------------------------------------------------------------------------- |
| ☁️ Deployment    | Host the backend and frontend on cloud platforms                                       |
| 📦 Model storage | Set up reliable remote model-artifact storage                                          |
| 🔐 Security      | Add authentication, file-size limits, and robust input validation before public access |
| ⚡ Performance   | Benchmark and optimize larger batch requests                                           |
| 🎯 Thresholding  | Explore thresholds based on validation data and application requirements               |
| 📊 Monitoring    | Add prediction monitoring and data-drift checks                                        |
| 🧪 Testing       | Expand automated inference and API tests                                               |
| 🔄 CI/CD         | Add automated testing and deployment workflows                                         |

> ⚠️ This is a machine learning portfolio and experimentation project. It is not a production-validated financial fraud prevention system.

## 👨‍💻 Author

**Deeptangshu Ghosh**
B.Tech — Computer Science & Engineering

🔗 GitHub: [Deeptangshu-008](https://github.com/Deeptangshu-008)

---

⭐ If you find this project interesting, feel free to explore the notebooks and star the repository.
