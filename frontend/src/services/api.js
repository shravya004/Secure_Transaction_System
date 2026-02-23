import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: BASE, timeout: 15000 });

// ── Helper: parse the message string from /transaction ────────────────────
// Backend returns strings like:
//   "✅ Transaction Approved — Risk Score: 0.23"
//   "🚨 Transaction Rejected by AI — Risk Score: 0.87"
//   "❌ Signature Invalid — Transaction Rejected"
function parseTransactionResult(message) {
  // Extract risk score from message if present
  const scoreMatch = message.match(/Risk Score[:\s]+([\d.]+)/i);
  const riskScore = scoreMatch ? parseFloat(scoreMatch[1]) : null;

  let status = 'REVIEW';
  let approved = false;

  if (message.includes('Approved') || message.includes('✅')) {
    status = 'APPROVED';
    approved = true;
  } else if (
    message.includes('Rejected') ||
    message.includes('🚨') ||
    message.includes('❌') ||
    message.includes('Invalid')
  ) {
    status = 'FLAGGED';
    approved = false;
  }

  return { status, riskScore, approved, message };
}

// ── Helper: parse blockchain blocks into frontend-friendly format ─────────
function normalizeBlock(block, index) {
  // block shape will be confirmed once we see blockchain.py
  // Common fields: index, timestamp, data/transactions, hash, previous_hash, nonce
  const tx = block.data?.transaction || block.transactions?.[0] || block.data || {};
  const enriched = block.data || {};

  return {
    id: block.hash || `block-${index}`,
    block_index: block.index ?? index,
    amount: tx.amount ?? enriched?.transaction?.amount ?? 0,
    sender: tx.sender ?? enriched?.transaction?.sender ?? '—',
    receiver: tx.receiver ?? enriched?.transaction?.receiver ?? '—',
    risk_score: enriched.risk_score ?? null,
    status: enriched.risk_score != null
      ? (enriched.risk_score > 0.5 ? 'FLAGGED' : 'APPROVED')
      : 'APPROVED',
    block_hash: block.hash ?? '—',
    prev_hash: block.previous_hash ?? block.prev_hash ?? '—',
    nonce: block.nonce ?? '—',
    signature: enriched.signature ?? '—',
    timestamp: block.timestamp
      ? new Date(block.timestamp * 1000).toISOString()
      : new Date().toISOString(),
  };
}

// ── 1. Submit transaction → POST /transaction ─────────────────────────────
export const predictTransaction = async (formData) => {
  // Map frontend form fields to backend's expected shape:
  // { features: [...], amount, sender, receiver }
  const features = [
    parseFloat(formData.v1) || 0,
    parseFloat(formData.v2) || 0,
    parseFloat(formData.v3) || 0,
    parseFloat(formData.v4) || 0,
    parseFloat(formData.v5) || 0,
    parseFloat(formData.v6) || 0,
    parseFloat(formData.time) || 0,
    parseFloat(formData.amount) || 0,
  ];

  const payload = {
    features,
    amount: parseFloat(formData.amount),
    sender: formData.user_id || 'user_unknown',
    receiver: formData.receiver || 'merchant_001',
  };

  const res = await api.post('/transaction', payload);
  const { status: httpStatus, message } = res.data;

  if (httpStatus !== 'success') {
    throw new Error(message || 'Transaction failed');
  }

  const parsed = parseTransactionResult(message);

  // Return in the shape the frontend UI expects
  return {
    status: parsed.status,
    risk_score: parsed.riskScore ?? (parsed.status === 'FLAGGED' ? 0.85 : 0.12),
    ai_score: parsed.riskScore ?? 0,
    behavior_score: 0, // not provided by this backend
    transaction_id: `tx-${Date.now()}`,
    raw_message: message,
    // SHAP not available from this backend — return null
    top_features: null,
  };
};

// ── 2. Get blockchain → GET /blockchain ───────────────────────────────────
export const getLedger = async () => {
  const res = await api.get('/blockchain');
  const { chain } = res.data;

  if (!Array.isArray(chain)) return [];

  // Skip genesis block (index 0) — it has no real transaction data
  return chain
    .slice(1)
    .map((block, i) => normalizeBlock(block, i + 1))
    .reverse(); // newest first
};

// ── 3. Dashboard stats — derived from blockchain data ─────────────────────
export const getDashboardStats = async () => {
  const res = await api.get('/blockchain');
  const { chain, chain_length, is_valid } = res.data;

  if (!Array.isArray(chain) || chain.length <= 1) {
    return {
      total_today: 0,
      flagged_count: 0,
      fraud_rate: 0,
      avg_risk: 0,
      status_counts: { APPROVED: 0, FLAGGED: 0 },
      chain_valid: is_valid,
      hourly_trend: [],
    };
  }

  const blocks = chain.slice(1); // skip genesis
  const total = blocks.length;

  const riskScores = blocks
    .map(b => b.data?.risk_score)
    .filter(s => s != null);

  const avgRisk = riskScores.length
    ? riskScores.reduce((a, b) => a + b, 0) / riskScores.length
    : 0;

  // Count flagged vs approved based on risk score threshold
  const flagged = riskScores.filter(s => s > 0.5).length;
  const approved = total - flagged;

  // Build hourly trend (group blocks by hour based on timestamp)
  const hourlyMap = {};
  blocks.forEach(block => {
    const ts = block.timestamp
      ? new Date(block.timestamp * 1000)
      : new Date();
    const hour = ts.getHours().toString().padStart(2, '0');
    if (!hourlyMap[hour]) hourlyMap[hour] = { hour, total: 0, flagged: 0 };
    hourlyMap[hour].total++;
    if (block.data?.risk_score > 0.5) hourlyMap[hour].flagged++;
  });
  const hourlyTrend = Object.values(hourlyMap).sort((a, b) =>
    a.hour.localeCompare(b.hour)
  );

  return {
    total_today: total,
    flagged_count: flagged,
    fraud_rate: total > 0 ? flagged / total : 0,
    avg_risk: avgRisk,
    status_counts: { APPROVED: approved, FLAGGED: flagged },
    chain_valid: is_valid,
    chain_length,
    hourly_trend: hourlyTrend,
    threshold_level: avgRisk > 0.4 ? 'tight' : 'normal',
  };
};

// ── 4. Recent transactions — last 10 blocks ───────────────────────────────
export const getRecentTransactions = async () => {
  const ledger = await getLedger();
  return ledger.slice(0, 10);
};

// ── 5. Alerts — flagged blocks with risk_score > 0.5 ─────────────────────
export const getAlerts = async () => {
  const ledger = await getLedger();
  return ledger
    .filter(tx => tx.status === 'FLAGGED')
    .map((tx, i) => ({
      ...tx,
      id: tx.id || `alert-${i}`,
      // Backend doesn't have an acknowledge endpoint yet —
      // we manage acknowledge state locally in the Alerts component
      top_features: null,
    }));
};

// ── 6. Acknowledge alert — no backend endpoint yet ───────────────────────
// Acknowledgement is handled client-side (localAcked state in Alerts.jsx)
// When Person 1 adds a backend endpoint, update this:
export const acknowledgeAlert = async (alertId) => {
  // TODO: replace with real endpoint when available
  // await api.post(`/alerts/${alertId}/acknowledge`);
  return { acknowledged: true, id: alertId };
};

// ── 7. System status — derived from blockchain health ────────────────────
export const getSystemStatus = async () => {
  const stats = await getDashboardStats();
  return {
    threshold_level: stats.threshold_level,
    threshold_value: stats.avg_risk,
    chain_valid: stats.chain_valid,
  };
};

// ── 8. Explain transaction — not available in this backend ───────────────
export const explainTransaction = async (_txId) => {
  // SHAP explanation not implemented in this backend
  return { top_features: null };
};