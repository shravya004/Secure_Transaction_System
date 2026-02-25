# 🛡️ FRAUD-X: Secure Transaction System

> AI + Blockchain + Cybersecurity fraud detection dashboard — based on the FRAUD-X framework (IEEE Access 2025)

![Stack](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-blue)
![Stack](https://img.shields.io/badge/Backend-FastAPI%20%2B%20Python-green)
![Stack](https://img.shields.io/badge/ML-DNN%20%2B%20SHAP-orange)
![Stack](https://img.shields.io/badge/Blockchain-Hyperledger%20Fabric-purple)

---

## 📋 Overview

The Secure Transaction System is a production-grade BFSI (Banking, Financial Services, and Insurance) fraud detection dashboard. It implements the **FRAUD-X multi-layer synergy pipeline**:

1. **AI-Based Detection** — Deep Neural Network (DNN) + Isolation Forest for anomaly scoring
2. **Blockchain Ledger** — Immutable transaction log with ECDSA signature verification (PBFT consensus)
3. **Cybersecurity Integration** — Intrusion detection logs correlated with transaction data
4. **Early Warning System** — Real-time adaptive threshold that tightens automatically on fraud spikes

**Results (IEEE Access 2025):** FRAUD-X achieves **99.5% accuracy**, **85.9% F1-score**, and **AUC 0.99** on the Credit Card Fraud dataset — outperforming single-plane AI models.

---

## 🏗️ System Architecture

```
User Browser (React/Vite)
        |
        | HTTP (Axios)
        ▼
  FastAPI Backend
  ├── /predict    → [DNN Model] + [Isolation Forest] + [Behavioral Profiler]
  │               → Multi-layer Risk Score → Adaptive Threshold Check
  ├── /explain    → [SHAP Explainer] → Top 5 Feature Importances
  ├── /ledger     → [Blockchain Ledger] → Block List
  ├── /alerts     → [Alert Store] → Active Alerts
  └── /system/status → Current threshold level
        |
        ▼
  Blockchain (Hyperledger Fabric / PBFT Consensus)
  └── Immutable Transaction Log → ECDSA Signature Verification
```

---

## 🚀 Quick Start (Docker — Recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### One-command launch

```bash
# Clone the repo
git clone https://github.com/shravya004/Secure_Transaction_System.git
cd Secure_Transaction_System

# Start everything
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Stop
```bash
docker compose down
```

---

## 💻 Local Development (Without Docker)

### Backend (Person 1)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Person 2)
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at http://localhost:5173

---

## 📁 Project Structure

```
Secure_Transaction_System/
├── backend/                   # FastAPI backend (Person 1)
│   ├── app/
│   │   └── main.py            # API routes: /predict, /explain, /ledger, /alerts
│   ├── ml/
│   │   └── fraud_model.pt     # Trained DNN model
│   └── Dockerfile
│
├── frontend/                  # React frontend (Person 2)
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/        # Sidebar, TopBar, Layout
│   │   │   └── ui/            # RiskGauge, ShapChart, StatCard, StatusBadge
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx  # Live stats + charts (auto-refresh 15s)
│   │   │   ├── Transaction.jsx# Submit form + risk gauge + SHAP
│   │   │   ├── Ledger.jsx     # Blockchain explorer with expandable rows
│   │   │   └── Alerts.jsx     # Real-time fraud alerts + acknowledge
│   │   ├── hooks/
│   │   │   ├── usePolling.js  # Auto-refresh hook
│   │   │   └── useTheme.js    # Dark mode toggle
│   │   └── services/
│   │       └── api.js         # All backend API calls (single source of truth)
│   ├── nginx.conf             # Production Nginx config
│   └── Dockerfile
│
└── docker-compose.yml         # One-command full-stack deployment
```

---

## 🖥️ Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/dashboard` | 4 stat cards, fraud trend line chart, status donut chart, recent activity table. Auto-refreshes every 15s. |
| Transaction | `/transaction` | Submit a transaction for fraud analysis. Shows risk gauge, APPROVED/FLAGGED/REVIEW badge, SHAP feature chart. |
| Ledger | `/ledger` | Blockchain explorer — searchable/filterable table. Click row to expand block hash, previous hash, nonce, signature. |
| Alerts | `/alerts` | Real-time flagged transactions. Acknowledge button, SHAP explanation, Early Warning System banner. |

---

## 🔌 Backend API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Submit transaction, returns `risk_score`, `status`, `ai_score`, `behavior_score`, `top_features` |
| `GET` | `/explain/{tx_id}` | SHAP explanation for a transaction |
| `GET` | `/ledger` | Full blockchain ledger |
| `GET` | `/transactions/recent` | Last 10 transactions |
| `GET` | `/dashboard/stats` | Aggregated stats for dashboard |
| `GET` | `/alerts` | Active fraud alerts |
| `POST` | `/alerts/{id}/acknowledge` | Acknowledge an alert |
| `GET` | `/system/status` | Current adaptive threshold level |

---

## 🎨 UI Features

- **Dark mode** — toggle in the top bar, persisted to localStorage, no flash on reload
- **Mobile responsive** — works at 375px, 768px, 1280px — tested at all breakpoints
- **Skeleton loaders** — no blank states while data loads
- **Page animations** — smooth fade-slide on every route transition
- **SHAP visualization** — red/blue horizontal bar chart showing which features drove the fraud score
- **Adaptive threshold banner** — automatically appears when backend raises the alert level

---

## ⚙️ Environment Variables

### `frontend/.env`
```env
# Development
VITE_API_URL=http://localhost:8000

# Production (Docker — Nginx proxies /api/ to backend)
# VITE_API_URL=/api
```

---

## 👥 Team

| Person | Role | 
|--------|------|
| Person 1 (shravya004) | FastAPI, blockchain, cybersecurity integration |
| Person 2 (tanishasenapati13) | Frontend (React dashboard, transaction UI, ledger explorer, alerts panel, SHAP) |
| Person 3 (Ishita2005cse) | Backend + ML, DNN model |


---

## 📄 Reference

B. Fetaji et al., *"FRAUD-X: An Integrated AI, Blockchain, and Cybersecurity Framework for Financial Transaction Security"*, IEEE Access, vol. 13, pp. 48068–48082, 2025.
