# 🤖 Multi-Agent Data Intelligence System

An intelligent web application that automatically cleans, preprocesses, and analyses any uploaded dataset using 5 specialized AI agents — powered by Python, Streamlit, CrewAI, and the Groq API.

---

## 📌 What This Project Does

Real-world data is always messy. It has missing values, duplicate entries, wrong formats, and irrelevant information. Before building any machine learning model, you first have to clean and preprocess the data — a process that can take up to 80% of a data scientist's total time.

This system solves that problem. You simply upload a CSV or Excel file, click a button, and the AI agents handle everything automatically in minutes. You can then download the clean, ready-to-use dataset.

---

## ✨ Key Features

- 📂 Upload any CSV or Excel file (up to 200MB)
- 🧹 Automatic data cleaning — removes duplicates, fills missing values, handles outliers
- ⚙️ Smart preprocessing — encodes text columns, scales numeric columns to 0–1
- 📊 Interactive analysis dashboard with 6 types of charts
- 💬 Conversational chatbot — ask questions about your data in plain English
- ⬇️ Download the processed dataset as a CSV file at any stage
- 🔄 Reset pipeline to restore the original dataset anytime
- 📍 Pipeline status tracker showing which steps are complete

---

## 🧠 The 5 AI Agents

| Agent | File | What It Does |
|---|---|---|
| Supervisor Agent | `agents/supervisor.py` | Reads user intent and routes to the correct agent |
| Cleaning Agent | `agents/cleaning_agent.py` | Removes duplicates, fills missing values, removes outliers |
| Preprocessing Agent | `agents/preprocessing_agent.py` | Encodes categories, scales numeric columns |
| Analysis Agent | `agents/analysis_agent.py` | Generates statistics, correlations, and interactive charts |
| Chatbot Agent | `agents/chatbot_agent.py` | Answers questions about the data in plain English |

---

## 🗂️ Project Structure

```
multi_agent_data_system/
│
├── app.py                        ← Main Streamlit web application
├── crew.py                       ← Central orchestrator connecting all agents
├── .env                          ← API key storage (never share this)
├── .gitignore                    ← Tells Git what not to upload
├── requirements.txt              ← All required Python libraries
│
├── agents/
│   ├── supervisor.py             ← Routes user requests to correct agent
│   ├── cleaning_agent.py         ← Data cleaning logic
│   ├── preprocessing_agent.py    ← Data preprocessing logic
│   ├── analysis_agent.py         ← Statistical analysis and chart generation
│   └── chatbot_agent.py          ← Conversational AI agent
│
├── utils/
│   ├── llm_config.py             ← Shared LLM configuration and API client
│   └── file_handler.py           ← File reading helper functions
│
├── datasets/                     ← Store sample CSV files here for testing
└── outputs/                      ← Processed files are saved here
```

---

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11.x | Core programming language |
| Streamlit | 1.56.0 | Web UI framework |
| CrewAI | 1.14.1 | Multi-agent orchestration |
| Groq API | 1.1.2 | LLM API access |
| LLaMA 3.1 8B | Instant | AI language model |
| Pandas | 3.0.2 | Data manipulation and processing |
| Scikit-learn | 1.8.0 | Encoding and scaling tools |
| Plotly | 6.6.0 | Interactive charts and graphs |
| NumPy | 2.4.4 | Numerical operations |
| Statsmodels | latest | OLS trendline for scatter plots |
| python-dotenv | 1.1.1 | Secure API key loading |
| openpyxl | 3.1.5 | Excel file support |

---

## ⚙️ Installation and Setup

### Step 1 — Prerequisites

Make sure you have the following installed:
- Python 3.11 from [python.org](https://www.python.org/downloads/) — tick **Add Python to PATH** during install
- Visual Studio Code from [code.visualstudio.com](https://code.visualstudio.com/)
- Git from [git-scm.com](https://git-scm.com/)

---

### Step 2 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-data-intelligence.git
cd multi-agent-data-intelligence
```

---

### Step 3 — Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line.

---

### Step 4 — Install All Required Libraries

```bash
pip install -r requirements.txt
```

This installs all libraries automatically. It takes 2–5 minutes.

---

### Step 5 — Get Your Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Click **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key — it starts with `gsk_...`

---

### Step 6 — Configure the API Key

Create a file called `.env` in the project root folder and add:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Replace the value with your actual Groq API key.

> ⚠️ Never share your `.env` file or upload it to GitHub. It is already listed in `.gitignore` for protection.

---

### Step 7 — Run the Application

```bash
streamlit run app.py
```

Your browser will automatically open at:
```
http://localhost:8501
```

---

## 🚀 How to Use the App

1. **Upload a file** — Click the upload area and choose any CSV or Excel file
2. **View the preview** — See your dataset's rows, columns, and missing value count
3. **Click Clean Data** — The Cleaning Agent removes duplicates, fills missing values, and handles outliers
4. **Click Preprocess** — The Preprocessing Agent encodes text columns and scales numeric columns
5. **Click Analyse** — The Analysis Agent generates statistics and charts
6. **Chat with your data** — Type any question in the chat box on the right
7. **Download** — Click the Download button to save your processed dataset as CSV
8. **Reset** — Click Reset Pipeline to restore the original dataset at any time

---

## 📊 Charts Available in the Analysis Dashboard

| Chart | What It Shows |
|---|---|
| Correlation Heatmap | How strongly each pair of numeric columns are related |
| Distribution Histograms | How values are spread across each numeric column |
| Box Plots | Median, spread, and outliers for each column |
| Scatter Plot with Trend Line | Relationship between any two numeric columns |
| Bar Chart | Frequency of each category in a text column |
| Donut / Pie Chart | Proportion of each category as a percentage |

---

## 🧪 Testing the System

### Quick Test — Create a Sample Dataset

```bash
python -c "
import pandas as pd
data = {
    'name':       ['Alice','Bob','Charlie','Bob','Eve',None,'Frank'],
    'age':        [25, 30, None, 30, 22, 999, 28],
    'salary':     [50000, 60000, 55000, 60000, None, 52000, 58000],
    'department': ['HR','IT','Finance','IT','HR',None,'IT'],
    'gender':     ['Female','Male','Male','Male','Female','Female','Male']
}
pd.DataFrame(data).to_csv('datasets/test_data.csv', index=False)
print('Test file created in datasets/ folder!')
"
```

Upload `datasets/test_data.csv` in the app to test all features.

### Test Each Agent Individually

```bash
# Test Supervisor Agent
python agents/supervisor.py

# Test Cleaning Agent
python agents/cleaning_agent.py

# Test Preprocessing Agent
python agents/preprocessing_agent.py

# Test Analysis Agent
python agents/analysis_agent.py

# Test Chatbot Agent
python agents/chatbot_agent.py

# Test Full Pipeline
python crew.py
```

### Real-World Datasets Used for Testing

- 🏠 House Price Prediction — [Kaggle](https://www.kaggle.com)
- 🎬 Netflix Dataset — [Kaggle](https://www.kaggle.com)
- 🇺🇸 US Government Data — [data.gov](https://www.data.gov)
- 🇮🇳 Indian Government Data — [data.gov.in](https://www.data.gov.in)

---

## 🔁 How the Routing Works

Every user request goes through this flow:

```
User types or clicks
        ↓
crew.py receives the request
        ↓
Supervisor Agent reads the request using LLaMA
        ↓
Returns one word: cleaning / preprocessing / analysis / chatbot
        ↓
crew.py sends to the correct agent
        ↓
Agent processes the dataset and returns result
        ↓
Result shown in the UI with AI summary
```

---

## ⚠️ Known Limitations

- Supports only CSV and Excel files — no JSON, SQL, or Parquet support yet
- Maximum file size is 200MB (Streamlit upload limit)
- IQR outlier removal may be aggressive on heavily skewed datasets (e.g. income data)
- Ordinal column encoding is rule-based and may not handle all column naming conventions
- Requires an internet connection — the Groq API does not work offline
- Ambiguous requests (e.g. "analyse and clean") may only route to one agent

---

## 🔮 Future Scope

- [ ] Support for more file types — JSON, SQL, Parquet
- [ ] AutoML model recommendations based on dataset type
- [ ] PDF report generation for download
- [ ] Cloud deployment with public URL via Streamlit Cloud
- [ ] Natural language querying of datasets
- [ ] Database integration for saving processed datasets
- [ ] User authentication with login and account management
- [ ] Advanced dashboard with custom chart builder

---

## 📁 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key starting with `gsk_...` |

---

## 🤝 Team and Acknowledgements

This project was built as a Semester VI Mini Project.

- Built using [Claude AI](https://claude.ai) for code generation and problem solving
- AI model powered by [LLaMA 3.1](https://ai.meta.com/llama/) via [Groq](https://groq.com)
- Multi-agent framework by [CrewAI](https://crewai.com)
- UI built with [Streamlit](https://streamlit.io)
- Concepts referenced from [GeeksforGeeks](https://www.geeksforgeeks.org)

---

## 📄 License

This project was created for educational purposes as part of a college mini project submission.

---

## 📬 Contact

For any queries about this project, please reach out through GitHub Issues.

---

*Multi-Agent Data Intelligence System — Semester VI Mini Project*