🤖 AI Data Analyst Chatbot: Conversational Insights from Structured Data

---

📌 Project Description

This project is an AI-powered data analyst chatbot that enables users to explore, analyze, and understand structured datasets using natural language. Built for analysts and data practitioners, it allows users to upload CSV or Excel files and ask questions in plain English to receive statistical insights, explanations, and visualizations in real time.

By combining automated data profiling, statistical analysis, and large language models, the chatbot bridges the gap between raw data and actionable insights without requiring SQL or extensive coding knowledge.

---

❓ Key Capabilities & Use Cases

📊 What columns, distributions, and summary statistics exist in the dataset?

🧹 Are there missing values, duplicates, or data quality issues?

📈 Which variables show the highest variation or contain outliers?

🔗 Are there correlations or patterns worth investigating?

🧪 Which statistical tests are appropriate for comparing groups or variables?

---

📂 Supported Data Inputs

The chatbot currently supports:
- CSV files
- Excel files (XLSX)
- Uploaded data is processed in-session and is not permanently stored.
  
---

🛠️ Technologies Used

This project leverages modern Python-based data and AI tooling:
- Python
- Streamlit (interactive UI)
- pandas & NumPy (data processing)
- SciPy (statistical analysis)
- Plotly (interactive visualisations)
- LangChain (v1) (LLM orchestration)
- Groq LLM (LLaMA 3.1) for fast, free inference

---

▶️ How to Run the Application

1️⃣ Clone the Repository
- git clone <repository-url>
- cd AI-Powered-Data-Analyst-Chatbot

2️⃣ Set Up a Virtual Environment
- python -m venv venv
- venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
- pip install -r requirements.txt

4️⃣ Set Environment Variable (Groq API Key)
- setx GROQ_API_KEY "your_api_key_here"


Restart your terminal after setting the key.

5️⃣ Run the App
- streamlit run src/app.py

The chatbot will be available at http://localhost:8501.

---

✅ Key Outcomes
- Enables conversational data exploration without SQL or notebooks
- Automatically profiles datasets and flags quality issues
- Provides statistical reasoning and recommendations
- Separates analytical reasoning from visualization for better performance
- Demonstrates real-world integration of LLMs with data pipelines

---

🤝 Contributing
Contributions are welcome. Feel free to fork the repository, submit pull requests, or open issues for enhancements or bug fixes.

---

📄 License
This project is open-source and available under the MIT License.

---
