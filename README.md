# 🧠 Entity Resolution Pipeline for Financial Customers

This repository contains an interactive and modular **Entity Resolution (ER)** system built using Python and Streamlit, designed for identifying and resolving duplicate or similar records across datasets—ideal for banking and financial customer data.

---

## ✨ Features

- 📥 File upload interface to compare two datasets (or use defaults)
- 🧱 Blocking logic to reduce pairwise comparisons
- ✅ Rule-based matching with fuzzy logic scoring
- 🤖 Machine learning matching using Random Forest
- 📊 Precision, recall, and confusion matrix visualizations
- 🕵️ Manual review interface for uncertain matches
- 💾 Save matched results to local CSVs
- 🚀 CI/CD pipeline with GitHub Actions
- 🔔 Slack alerts on build failures (optional)

---

## 📁 Project Structure

```
banking_er_pipeline/
├── app/
│   └── main.py              # Streamlit UI and pipeline orchestration
├── data/
│   ├── raw/                 # Input datasets (source1.csv, source2.csv)
│   └── processed/           # Output match results
├── deployment/
│   └── requirements.txt     # All Python dependencies
├── .github/workflows/
│   └── ci-cd.yml            # GitHub Actions CI/CD workflow
├── tests/
│   └── test_basic.py        # Import and environment test
└── README.md
```

---

## 🗃️ Input File Format

Both CSV files should have the following columns:

| Column       | Description                     |
|--------------|---------------------------------|
| customer_id  | Unique customer ID              |
| full_name    | Full customer name              |
| last_name    | Last name (used for blocking)   |
| address      | Address line                    |
| zip          | ZIP code                        |
| email        | Email address                   |

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/banking_er_pipeline.git
cd banking_er_pipeline
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r deployment/requirements.txt
streamlit run app/main.py
```

---

## ⚙️ CI/CD with GitHub Actions

On every push to `main`, this project:
- Sets up a Python environment
- Installs all required packages
- Runs a basic test script to ensure dependencies are installed
- (Optionally) Sends a Slack notification if the build fails

---

## 📅 Changelog Snapshot

See full details in `CHANGELOG.md`.

```
## [1.0.1] - 2025-04-18
- Add CI/CD via GitHub Actions
- Add file upload support in Streamlit
- Add Slack notification integration
- Add confidence scoring & review UI
```

---

## 🙌 Credits

Built with ❤️ by Prashant Tyagi. PRs and forks welcome!


