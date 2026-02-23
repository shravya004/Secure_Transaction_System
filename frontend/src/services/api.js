import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: BASE, timeout: 15000 });

// ─────────────────────────────────────────────────────────────────────────────
// BACKEND CONTRACT (from reading the actual source code)
//
// POST /transaction
//   Request:  { features: float[], amount: float, sender: str, receiver: str }
//   Response: { status: "success", message: "✅ Transaction Approved — Risk Score: 0.23" }
//             { status: "success", message: "🚨 Transaction Rejected by AI — Risk Score: 0.87" }
//             { status: "success", message: "❌ Signature Invalid — Transaction Rejected" }
//
// GET /blockchain
//   Response: {
//     chain_length: int,
//     is_valid: bool,
//     chain: [
//       { index: 0, timestamp: float, data: "Genesis Block", previous_hash: "0", nonce: 0, hash: str },
//       { index: 1, timestamp: float, data: { transaction: { features, amount, sender, receiver },
//                                             risk_score: float (0.0–1.0),
//                                             signature: str },
//         previous_hash: str, nonce: int, hash: str }
//     ]
//   }
// ─────────────────────────────────────────────────────────────────────────────

// Parse message string → { status, riskScore, approved }
function parseMessage(message) {
  const scoreMatch = message.match(/Risk Score[:\s]+([\d.]+)/i);
  const riskScore  = scoreMatch ? parseFloat(scoreMatch[1]) : null;

  let status = 'REVIEW';
  if (message.includes('Approved') || message.includes('✅')) {
    status = 'APPROVED';
  } else if (
    message.includes('Rejected') ||
    message.includes('🚨') ||
    message.includes('❌') ||
    message.includes('Invalid')
  ) {
    status = 'FLAGGED';
  }

  return { status, riskScore };
}

// Convert a real blockchain block → frontend-friendly shape
// Block fields: { index, timestamp, data: { transaction, risk_score, signature },
//                 previous_hash, nonce, hash }
function normalizeBlock(block) {
  const tx  = block.data?.transaction || {};
  const rs  = block.data?.risk_score;
  const sig = block.data?.signature;

  return {
    // identity
    id:          block.hash,
    block_index: block.index,

    // transaction payload
    amount:   tx.amount   ?? 0,
    sender:   tx.sender   ?? '—',
    receiver: tx.receiver ?? '—',
    features: tx.features ?? [],

    // risk
    risk_score: rs ?? null,
    status:     rs != null
                  ? (rs > 0.5 ? 'FLAGGED' : 'APPROVED')
                  : 'APPROVED',

    // blockchain proof
    block_hash:    block.hash,
    prev_hash:     block.previous_hash,
    nonce:         block.nonce,
    signature:     sig ?? '—',

    // time — backend stores Unix float seconds
    timestamp: new Date(block.timestamp * 1000).toISOString(),
  };
}

// ── 1. Submit transaction ─────────────────────────────────────────────────
export const predictTransaction = async (formData) => {
  // Map UI form → backend payload
  // features[] must be floats matching the scaler's expected input_dim
  const features = [
    parseFloat(formData.v1)  || 0,
    parseFloat(formData.v2)  || 0,
    parseFloat(formData.v3)  || 0,
    parseFloat(formData.v4)  || 0,
    parseFloat(formData.v5)  || 0,
    parseFloat(formData.v6)  || 0,
    parseFloat(formData.time)   || 0,
    parseFloat(formData.amount) || 0,
  ];

  const payload = {
    features,
    amount:   parseFloat(formData.amount),
    sender:   formData.sender   || formData.user_id || 'user_unknown',
    receiver: formData.receiver || 'merchant_001',
  };

  const res = await api.post('/transaction', payload);

  if (res.data.status !== 'success') {
    throw new Error(res.data.message || 'Transaction failed');
  }

  const { status, riskScore } = parseMessage(res.data.message);

  return {
    status,
    risk_score:     riskScore ?? (status === 'FLAGGED' ? 0.82 : 0.11),
    ai_score:       riskScore ?? 0,
    behavior_score: 0,             // not in this backend
    transaction_id: `tx-${Date.now()}`,
    raw_message:    res.data.message,
    top_features:   null,          // SHAP not in this backend
  };
};

// ── 2. Get full blockchain ────────────────────────────────────────────────
export const getLedger = async () => {
  const res  = await api.get('/blockchain');
  const chain = res.data.chain ?? [];

  return chain
    .filter(b => b.index !== 0)          // skip genesis block
    .map(normalizeBlock)
    .reverse();                           // newest first
};

// ── 3. Dashboard stats — computed from blockchain ─────────────────────────
export const getDashboardStats = async () => {
  const res = await api.get('/blockchain');
  const { chain = [], chain_length = 0, is_valid = true } = res.data;

  const realBlocks = chain.filter(b => b.index !== 0);
  const total      = realBlocks.length;

  if (total === 0) {
    return {
      total_today:   0,
      flagged_count: 0,
      fraud_rate:    0,
      avg_risk:      0,
      status_counts: { APPROVED: 0, FLAGGED: 0 },
      chain_length,
      chain_valid:   is_valid,
      hourly_trend:  [],
      threshold_level: 'normal',
    };
  }

  const riskScores  = realBlocks.map(b => b.data?.risk_score).filter(s => s != null);
  const flaggedCount = riskScores.filter(s => s > 0.5).length;
  const approvedCount = total - flaggedCount;
  const avgRisk     = riskScores.length
    ? riskScores.reduce((a, b) => a + b, 0) / riskScores.length
    : 0;

  // Group by hour of day using Unix timestamp
  const hourlyMap = {};
  realBlocks.forEach(block => {
    const date = new Date(block.timestamp * 1000);
    const hour = date.getHours().toString().padStart(2, '0');
    if (!hourlyMap[hour]) hourlyMap[hour] = { hour, total: 0, flagged: 0 };
    hourlyMap[hour].total++;
    if ((block.data?.risk_score ?? 0) > 0.5) hourlyMap[hour].flagged++;
  });
  const hourlyTrend = Object.values(hourlyMap).sort((a, b) =>
    a.hour.localeCompare(b.hour)
  );

  return {
    total_today:     total,
    flagged_count:   flaggedCount,
    fraud_rate:      flaggedCount / total,
    avg_risk:        avgRisk,
    status_counts:   { APPROVED: approvedCount, FLAGGED: flaggedCount },
    chain_length,
    chain_valid:     is_valid,
    hourly_trend:    hourlyTrend,
    threshold_level: avgRisk > 0.4 ? 'tight' : 'normal',
  };
};

// ── 4. Recent transactions ────────────────────────────────────────────────
export const getRecentTransactions = async () => {
  const ledger = await getLedger();
  return ledger.slice(0, 10);
};

// ── 5. Alerts — blocks where risk_score > 0.5 ────────────────────────────
export const getAlerts = async () => {
  const ledger = await getLedger();
  return ledger
    .filter(tx => tx.status === 'FLAGGED')
    .map(tx => ({
      ...tx,
      top_features: null,   // SHAP not available in this backend
    }));
};

// ── 6. Acknowledge alert — client-side only (no backend endpoint) ─────────
// Alerts.jsx manages acknowledged IDs in React state (localAcked)
export const acknowledgeAlert = async (alertId) => {
  // When backend adds POST /alerts/{id}/acknowledge, replace with:
  // await api.post(`/alerts/${alertId}/acknowledge`);
  return { acknowledged: true, id: alertId };
};

// ── 7. System status — derived from chain health ──────────────────────────
export const getSystemStatus = async () => {
  const stats = await getDashboardStats();
  return {
    threshold_level: stats.threshold_level,
    threshold_value: stats.avg_risk,
    chain_valid:     stats.chain_valid,
  };
};

// ── 8. Explain transaction — not in this backend ──────────────────────────
export const explainTransaction = async (_txId) => {
  return { top_features: null };
};