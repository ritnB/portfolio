## 👋 Welcome! Thanks for stopping by

This repo is a hub for my AI and data projects. Some production details are anonymized for portfolio sharing, but each project is fully navigable and documented. For usage and setup, check the `README.md` inside each project folder.

## 🔎 Projects at a glance

- **Agentic AI**: An LLM-powered multi-agent system that plans → writes → evaluates content, calling tools for trend, sentiment, and prediction analysis to improve results. Includes an optional end-to-end flow up to Threads posting.
  - Tech: Python, LangChain, LLM providers, tool calling (trend/sentiment/prediction), lightweight app server
  - Docs: [Agentic AI/README.md](Agentic%20AI/README.md)

- **Crypto Prediction**: Transformer-based time series modeling for asset movement prediction with automated retraining and verification pipelines, real-time inference API, and GCS/Supabase integration.
  - Tech: Python, PyTorch, Flask, Supabase (PostgreSQL), GCS, Optuna, Docker
  - Docs: [Crypto Prediction/README.md](Crypto%20Prediction/README.md)

- **QNN Security**: Notebooks exploring Quantum Neural Networks for security traffic classification using datasets like UNSW-NB15.
  - Tech: Python, Jupyter Notebooks, quantum-circuit/QNN concept experiments
  - Docs: [QNN Security/README.md](QNN%20Security/README.md)

- **Trajectory-Based UAV Location Integrity Analysis**: Data pipelines and notebooks to detect integrity anomalies (spoofing/attacks) from drone trajectories and motion patterns.
  - Tech: Python, preprocessing/visualization, time-series/sequence builder notebooks
  - Folder: [Trajectory-Based UAV Location Integrity Analysis/](Trajectory-Based%20UAV%20Location%20Integrity%20Analysis/)

## 🗂️ Repo layout

```
portfolio/
├─ Agentic AI/
├─ Crypto Prediction/
├─ QNN Security/
└─ Trajectory-Based UAV Location Integrity Analysis/
```

Each folder includes a project-specific `README.md`, `requirements.txt`, and runnable scripts or notebooks.

## 🚀 Quick tour

1) Clone the repo
```bash
git clone <repository-url>
cd portfolio
```
2) Jump into any project and follow its `README.md`
```bash
cd "Crypto Prediction"
# or
cd "Agentic AI"
```
3) A Python virtual environment is recommended. Install deps from each folder’s `requirements.txt`.

## 🧰 Common toolbox

- **AI/ML**: PyTorch, scikit-learn, Transformers
- **LLM & Agents**: LangChain, prompt design, tool calling patterns
- **Backend/Service**: Python (Flask, etc.), lightweight APIs, Docker
- **Data/Cloud**: Pandas/NumPy, Google Cloud Storage, Supabase (PostgreSQL)
- **Experimentation**: Jupyter, preprocessing/analysis pipelines

## 🙌 What I care about

- A cohesive flow from **problem framing → data pipeline → model/agent design → ops automation**
- Production-ready practices: **configurability, observability/monitoring, automated retraining**
- Clear structure and documentation for smooth team collaboration

## 📬 Contact

Always happy to chat about projects or collaborations. Open an issue/PR or reach out via the profile links.

— Thanks for visiting. Browse freely, run lightly 😊
