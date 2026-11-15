# Zepto Data & AI Platform

## 1. Project Overview

The Zepto Data & AI Platform is a capstone project demonstrating data collection, data cleaning, database management, exploratory data analysis, machine learning, and an AI-powered customer support assistant.

The project contains three independent modules:

1. **Data Pipeline:** Collects book information, cleans the data, stores it in SQLite, and performs SQL and pandas analysis.
2. **Analytics:** Performs exploratory data analysis, visualization, classification, regression, and model evaluation using the Titanic dataset.
3. **Support Assistant:** Implements a Retrieval-Augmented Generation (RAG) application using Zepto policy documents, ChromaDB, LangGraph, and FastAPI.

## 2. Technologies Used

- Python 3.12
- Pandas and NumPy
- Requests and BeautifulSoup
- SQLite
- Matplotlib and Seaborn
- Scikit-learn and Imbalanced-learn
- Sentence Transformers
- ChromaDB
- LangGraph
- FastAPI and Pydantic
- Docker
- Groq (optional real LLM mode)

## 3. Project Structure

```text
Zepto_Data_AI_Platform/
│
├── data_pipeline/
│   ├── scraper.py
│   ├── clean_data.py
│   ├── database.py
│   ├── queries.py
│   ├── pandas_sql.py
│   ├── books_raw.csv
│   ├── books_clean.csv
│   ├── books.db
│   └── README.md
│
├── analytics/
│   ├── titanic_analysis.py
│   ├── cleaning.py
│   ├── eda.py
│   ├── survival_analysis.py
│   ├── correlation.py
│   ├── multivariate.py
│   ├── standardization.py
│   ├── classification.py
│   ├── class_imbalance.py
│   ├── model_tuning.py
│   ├── regression.py
│   ├── final_model.py
│   ├── final_comparison.py
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   ├── chroma_db/
│   ├── create_docs.py
│   ├── ingest.py
│   ├── prompts.py
│   ├── main.py
│   ├── api.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 4. Setup Instructions

### Prerequisites

- Python 3.12 or later
- Git
- Docker Desktop with WSL 2 integration (required for Docker deployment)

### Clone the Repository

```bash
git clone https://github.com/OnkarKolhe22/Zepto_Data_AI_Platform.git
cd Zepto_Data_AI_Platform
```

### Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Ubuntu/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

The project uses one consolidated root `requirements.txt` for all three modules.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `support_assistant/requirements.txt` file is also retained for installing Support Assistant dependencies independently.

## 5. How to Run the Modules

Run all commands below from the project root directory, unless stated otherwise.

### Module 1: Data Pipeline

The pipeline scrapes book information, cleans the data, creates the SQLite database, executes SQL queries, and compares SQL results with pandas operations.

Run the scripts in this order:

```bash
python data_pipeline/scraper.py
python data_pipeline/clean_data.py
python data_pipeline/database.py
python data_pipeline/queries.py
python data_pipeline/pandas_sql.py
```

**Expected outputs:**

- `books_raw.csv`
- `books_clean.csv`
- `books.db`
- `sql_query_results.txt`

The pipeline collects book information from the first five pages of Books to Scrape.

### Module 2: Analytics

The analytics module uses the Titanic dataset for data cleaning, exploratory analysis, visualization, classification, regression, and model evaluation.

Run the scripts in this order:

```bash
python analytics/titanic_analysis.py
python analytics/cleaning.py
python analytics/eda.py
python analytics/survival_analysis.py
python analytics/correlation.py
python analytics/multivariate.py
python analytics/standardization.py
python analytics/classification.py
python analytics/class_imbalance.py
python analytics/model_tuning.py
python analytics/regression.py
python analytics/final_model.py
python analytics/final_comparison.py
```

**Expected outputs include:**

- Original and cleaned Titanic datasets
- EDA charts and visualizations
- Classification and regression results
- Model comparison results
- `titanic_survival_pipeline.joblib`

The original dataset is saved locally as a CSV fallback.

### Module 3: Support Assistant

The Support Assistant uses eight policy documents, sentence-transformer embeddings, ChromaDB retrieval, LangGraph workflow orchestration, and FastAPI.

First, navigate to the module directory:

```bash
cd support_assistant
```

Install its dependencies independently if needed:

```bash
pip install -r requirements.txt
```

Run document ingestion:

```bash
python ingest.py
```

Start the API:

```bash
uvicorn api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

The main endpoint is:

`POST /ask`

The default configuration uses `MOCK_LLM=1`, which supports deterministic offline testing while still performing document retrieval for policy questions.

For optional real LLM mode, configure `MOCK_LLM=0` and provide a valid `GROQ_API_KEY` environment variable.

### Docker Deployment (Optional)

Run these commands from the `support_assistant` directory.

Build the Docker image:

```bash
docker build -t zepto-support-assistant .
```

Start the container:

```bash
docker run -d --name zepto-support -p 8000:8000 -e MOCK_LLM=1 zepto-support-assistant
```

Open:

http://localhost:8000/docs

Check running containers:

```bash
docker ps
```

Stop the container:

```bash
docker stop zepto-support
```

## 6. Design Decisions

### Module 1: Data Pipeline

- Used BeautifulSoup to collect structured book information from the website.
- Used pandas to clean prices, ratings, availability, and missing values.
- Converted GBP prices to INR using a fixed exchange rate of 105.50 INR per GBP.
- Designed a normalized SQLite database with separate categories and books tables to reduce data duplication.
- Used primary and foreign keys to maintain database relationships.
- Compared SQL JOIN results with pandas merge results to validate consistency.

### Module 2: Analytics

- Used the Titanic dataset to demonstrate a complete analytics and machine learning workflow.
- Used median imputation for missing age values and removed rows with missing embarkation information.
- Dropped the deck column because it contained a high proportion of missing values.
- Used visualizations and statistical measures to understand distributions, outliers, correlations, and survival patterns.
- Compared Logistic Regression, Decision Tree, and Random Forest classifiers.
- Used stratified train/test splitting to preserve class proportions.
- Compared baseline, class-weighted, and SMOTE approaches to investigate class imbalance.
- Used GridSearchCV and out-of-bag evaluation for Random Forest tuning.
- Saved the final classification pipeline using joblib for reuse and testing.
- Evaluated fare prediction using regression metrics and residual analysis.

### Module 3: Support Assistant

- Used eight policy documents as the knowledge source for customer support questions.
- Used the all-MiniLM-L6-v2 sentence-transformer model to generate document and query embeddings.
- Used persistent ChromaDB storage with cosine similarity for semantic retrieval.
- Retrieved the three most relevant policy documents for policy questions.
- Used LangGraph to classify intent and route questions through different processing nodes.
- Used structured prompts to guide answer generation.
- Used Pydantic to validate API response structure.
- Used FastAPI to expose the assistant through a REST endpoint.
- Implemented a deterministic mock mode for offline testing and an optional Groq-based real LLM mode.
- Used Docker to support containerized deployment.

## 7. Testing and Results

### Module 1

- Successfully collected 100 books.
- Stored 100 book records across 29 categories.
- Executed SQL queries and saved their outputs.
- Reproduced SQL JOIN results using pandas merge.

### Module 2

- Processed the Titanic dataset from 891 original rows to 889 cleaned rows.
- Achieved 83.15% test accuracy with the tuned Random Forest classifier.
- Saved and successfully reloaded the final classification pipeline.
- Evaluated fare regression and documented its limitations.

### Module 3

- Successfully ingested eight policy documents into ChromaDB.
- Tested policy and general questions through the FastAPI endpoint.
- Verified valid HTTP 200 responses.
- Successfully built and tested the Docker deployment.

## 8. Git and Version Control

The project is maintained in a public GitHub repository:

https://github.com/OnkarKolhe22/Zepto_Data_AI_Platform

Git workflow included:

- Initial project setup.
- Feature branch creation: `setup-project-structure`.
- Multiple commits on the feature branch.
- Merging the feature branch into `main`.
- Separate commits for the three project modules and final documentation.

## 9. Project Status

All three capstone modules have been implemented, tested, documented, and uploaded to the public GitHub repository.

Each module also contains its own README with additional technical details.

## 10. Author

GitHub: https://github.com/OnkarKolhe22
