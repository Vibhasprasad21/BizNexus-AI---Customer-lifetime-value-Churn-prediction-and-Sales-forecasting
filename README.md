# 🚀 BizNexus AI

**BizNexus AI** is an advanced AI-driven business analytics platform that provides sophisticated insights into:

- 🧮 Customer Lifetime Value (CLV)
- ⚠️ Churn Prediction
- 📈 Sales Forecasting  
All in one unified application!

---

## 🌟 Features

- 🔍 **CLV Analysis**: Identify high-value customers with AI-powered predictions.
- 🔁 **Churn Prediction**: Spot at-risk customers and take action.
- 📊 **Sales Forecasting**: Forecast revenue with time-series models.
- 🤖 **AI Business Assistant**: Ask business questions in natural language.
- 📉 **Interactive Dashboards**: Dynamic charts for your business data.
- 🛎️ **Intelligent Alerts**: Get notified about potential risks via email.

---

## 📂 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
  - [Firebase Setup](#firebase-setup)
  - [Google Gemini API](#google-gemini-api)
  - [SMTP Setup](#smtp-setup)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🛠 Installation

### ✅ Prerequisites

- Python 3.8+
- `pip`
- Firebase account
- Google Cloud account (for Gemini API)

### 🔧 Setup

```bash
git clone https://github.com/yourusername/biznexus-ai.git
cd biznexus-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

⚙️ Configuration
🔥 Firebase Setup
Create a project at firebase.google.com

Enable:

Authentication (Email/Password)

Firestore

Generate a service account key.

Create .streamlit/secrets.toml with:

toml
Copy
Edit
[firebase.config]
apiKey = "YOUR_API_KEY"
authDomain = "YOUR_PROJECT_ID.firebaseapp.com"
projectId = "YOUR_PROJECT_ID"
...
🧠 Google Gemini API (for AI Assistant)
Get an API key from Google AI Studio

Add to secrets.toml:

toml
Copy
Edit
[gemini]
api_key = "YOUR_GEMINI_API_KEY"
📧 SMTP Setup (for Email Alerts)
toml
Copy
Edit
[email]
sender_email = "your-email@gmail.com"
smtp_server = "smtp.gmail.com"
smtp_port = 465
smtp_username = "your-email@gmail.com"
smtp_password = "your-app-password"
use_ssl = true
🔒 Use Gmail App Password if needed.

▶️ Usage
bash
Copy
Edit
streamlit run app.py
Visit: http://localhost:8501

🧑‍💼 Authentication
Sign up or log in.

Enter business details to complete profile.

📁 Data Upload
Upload your customer transaction CSV.

System processes & validates it.

📊 CLV Analysis
Run CLV predictions

Explore value tiers

View interactive visualizations

🚨 Churn Prediction
Identify high-risk customers

View churn probabilities

Get strategy recommendations

📈 Sales Forecasting
Predict sales trends

Analyze seasonal patterns

Simulate promo impacts

🤖 AI Assistant
Ask natural language questions like:

"Who are my top 10 customers by CLV?"

"What's the forecast for next month?"

🧾 Data Requirements

Field	Description
Customer ID	Unique identifier for each customer
Purchase Date	When the purchase happened
Transaction Amount	Value of the purchase
Product ID (opt)	What was bought
Additional fields like demographics, region, and marketing data improve results.

🧰 Technology Stack

Component	Technology
Frontend	Streamlit
Backend	Python, Firebase
ML Models	Gamma-Gamma (CLV), XGBoost (Churn), BiLSTM (Forecasting)
NLP Assistant	Google Gemini API
Visualization	Plotly
Alerts	SMTP Email
📁 Project Structure
bash
Copy
Edit
biznexus-ai/
├── app.py
├── pages/
│   ├── 01_Home.py
│   ├── 02_Authentication.py
│   ├── 03_Upload.py
│   ├── ...
├── src/
│   ├── auth/
│   ├── data_processing/
│   ├── models/
│   ├── firebase/
│   └── utils/
├── assets/
│   ├── images/
│   └── css/
├── requirements.txt
└── README.md
🤝 Contributing
Fork the repo

Create a new branch:

bash
Copy
Edit
git checkout -b feature-name
Commit and push:

bash
Copy
Edit
git commit -am "Add feature"
git push origin feature-name
Submit a Pull Request

