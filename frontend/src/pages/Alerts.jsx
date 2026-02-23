import { useState, useCallback } from 'react';
import { Bell, BellOff, AlertTriangle, CheckCircle2, ShieldAlert, Link2 } from 'lucide-react';
import StatusBadge from '../components/ui/StatusBadge';
import { usePolling } from '../hooks/usePolling';
import { getAlerts, getDashboardStats, acknowledgeAlert } from '../services/api';

function Skeleton({ className = '' }) {
  return <div className={`animate-pulse bg-gray-200 dark:bg-gray-700 rounded ${className}`} />;
}

export default function Alerts() {
  const [expanded,   setExpanded]   = useState(null);
  const [acking,     setAcking]     = useState(null);
  const [localAcked, setLocalAcked] = useState([]);

  const alertsFn = useCallback(() => getAlerts(),        []);
  const statsFn  = useCallback(() => getDashboardStats(), []);

  const { data: alertData, loading } = usePolling(alertsFn, 15000);
  const { data: stats }              = usePolling(statsFn,  30000);

  const allAlerts = alertData || [];
  const active    = allAlerts.filter(a => !localAcked.includes(a.id));
  const acked     = allAlerts.filter(a =>  localAcked.includes(a.id));

  const handleAck = async (alertId) => {
    setAcking(alertId);
    try {
      await acknowledgeAlert(alertId);
      setLocalAcked(prev => [...prev, alertId]);
      setExpanded(null);
    } catch (e) {
      console.error('Ack failed:', e);
    } finally {
      setAcking(null);
    }
  };

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Bell size={22} className="text-red-500" />
            Fraud Alerts
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
            Blockchain blocks where PyTorch risk score &gt; 0.5 — auto-refreshes every 15s
          </p>
        </div>
        {active.length > 0 && (
          <span className="flex items-center gap-1.5 bg-red-100 dark:bg-red-900/30
                           text-red-700 dark:text-red-300 px-3 py-1 rounded-full text-xs font-semibold">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            {active.length} Active
          </span>
        )}
      </div>

      {/* Early warning banner */}
      {stats?.threshold_level === 'tight' && (
        <div className="flex items-start gap-3 bg-red-50 dark:bg-red-900/20
                        border border-red-300 dark:border-red-700 rounded-xl px-4 py-3">
          <ShieldAlert size={18} className="text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-700 dark:text-red-300">
              Elevated Risk Detected
            </p>
            <p className="text-xs text-red-600 dark:text-red-400 mt-0.5">
              Average chain risk score is {(stats.avg_risk * 100).toFixed(1)}% — above the 40% threshold.
              {stats.flagged_count} of {stats.total_today} transactions flagged.
            </p>
          </div>
        </div>
      )}

      {/* Active alerts */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400
                       flex items-center gap-2">
          <AlertTriangle size={14} className="text-red-500" />
          Active ({loading ? '…' : active.length})
        </h2>

        {loading ? (
          <div className="space-y-3">
            {Array(3).fill(0).map((_, i) => <Skeleton key={i} className="h-20" />)}
          </div>
        ) : active.length === 0 ? (
          <div className="card flex flex-col items-center py-12 text-center">
            <BellOff size={32} className="text-gray-300 dark:text-gray-600 mb-3" />
            <p className="text-gray-500 font-medium">No active alerts</p>
            <p className="text-xs text-gray-400 mt-1">
              Flagged transactions (risk &gt; 0.5) from the blockchain will appear here.
            </p>
          </div>
        ) : (
          active.map((alert, i) => (
            <div key={alert.id || i}
                 className="card border-l-4 border-l-red-500 p-4 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <StatusBadge status={alert.status} />
                    <span className="font-semibold text-gray-900 dark:text-white">
                      ${Number(alert.amount).toFixed(2)}
                    </span>
                    <span className="text-xs text-red-600 dark:text-red-400 font-bold">
                      Risk: {(alert.risk_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 font-mono">
                    {alert.sender} → {alert.receiver}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Block #{alert.block_index} · {new Date(alert.timestamp).toLocaleString()}
                  </p>
                </div>

                <div className="flex gap-2 flex-shrink-0">
                  <button
                    onClick={() => setExpanded(expanded === i ? null : i)}
                    className="btn-secondary text-xs py-1.5"
                  >
                    {expanded === i ? 'Hide' : 'Block Details'}
                  </button>
                  <button
                    onClick={() => handleAck(alert.id)}
                    disabled={acking === alert.id}
                    className="btn-primary text-xs py-1.5 disabled:opacity-50 flex items-center gap-1"
                  >
                    {acking === alert.id
                      ? <><span className="w-3 h-3 border-2 border-white/50 border-t-white rounded-full animate-spin" />Acknowledging...</>
                      : 'Acknowledge'
                    }
                  </button>
                </div>
              </div>

              {/* Block detail expand */}
              {expanded === i && (
                <div className="border-t dark:border-gray-700 pt-3 grid grid-cols-1 md:grid-cols-2 gap-2 font-mono text-xs">
                  {[
                    ['Block Hash',    alert.block_hash],
                    ['Previous Hash', alert.prev_hash],
                    ['Nonce (PoW)',    alert.nonce],
                    ['Signature',     alert.signature !== '—' ? alert.signature?.slice(0, 40) + '...' : '—'],
                  ].map(([k, v]) => (
                    <div key={k} className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-2.5">
                      <span className="text-gray-400 block mb-0.5 text-xs">{k}</span>
                      <span className="text-blue-700 dark:text-blue-400 break-all">{v ?? '—'}</span>
                    </div>
                  ))}
                  <div className="col-span-full">
                    <p className="text-gray-400 text-xs">
                      Note: SHAP feature explanation is not available in the current backend.
                      The risk score comes directly from the PyTorch TrustScoreModel output.
                    </p>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Acknowledged */}
      {acked.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400 flex items-center gap-2">
            <CheckCircle2 size={14} className="text-green-500" />
            Acknowledged ({acked.length})
          </h2>
          <div className="space-y-2 opacity-60">
            {acked.slice(0, 5).map((alert, i) => (
              <div key={alert.id || i}
                   className="card py-3 px-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <CheckCircle2 size={14} className="text-green-500 flex-shrink-0" />
                  <span className="text-sm font-medium">${Number(alert.amount).toFixed(2)}</span>
                  <span className="font-mono text-xs text-gray-500">{alert.sender} → {alert.receiver}</span>
                </div>
                <span className="text-xs text-gray-400">Block #{alert.block_index}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Link to ledger */}
      <div className="flex items-center gap-2 text-xs text-gray-400 pt-2">
        <Link2 size={12} />
        All transactions (including approved) are visible in the Ledger page.
      </div>
    </div>
  );
}