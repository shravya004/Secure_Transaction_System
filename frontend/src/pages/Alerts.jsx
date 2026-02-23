import { useState, useCallback } from 'react';
import { Bell, BellOff, AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';
import ShapChart from '../components/ui/ShapChart';
import StatusBadge from '../components/ui/StatusBadge';
import { usePolling } from '../hooks/usePolling';
import { getAlerts, acknowledgeAlert, getSystemStatus } from '../services/api';

function Skeleton({ className = '' }) {
  return <div className={`animate-pulse bg-gray-200 dark:bg-gray-700 rounded ${className}`} />;
}

export default function Alerts() {
  const [expanded, setExpanded] = useState(null);
  const [acking, setAcking]     = useState(null);
  const [localAcked, setLocalAcked] = useState([]);

  const alertsFn = useCallback(() => getAlerts(), []);
  const statusFn = useCallback(() => getSystemStatus(), []);

  const { data: alertData, loading, refetch } = usePolling(alertsFn, 15000);
  const { data: sysStatus }                    = usePolling(statusFn, 30000);

  const allAlerts = alertData || [];

  // Split into active vs acknowledged
  const active = allAlerts.filter(a =>
    a.status !== 'ACKNOWLEDGED' && !localAcked.includes(a.id)
  );
  const acked = [
    ...allAlerts.filter(a => a.status === 'ACKNOWLEDGED'),
    ...allAlerts.filter(a => localAcked.includes(a.id)),
  ];

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

  // Mask card number helper
  const maskCard = (str) =>
    str ? str.replace(/(\d{4})\d+(\d{4})/, '$1****$2') : str;

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
            Real-time flagged transactions — auto-refreshes every 15s
          </p>
        </div>
        <div className="flex items-center gap-3">
          {active.length > 0 && (
            <span className="flex items-center gap-1.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-3 py-1 rounded-full text-xs font-semibold">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              {active.length} Active
            </span>
          )}
        </div>
      </div>

      {/* Early Warning System Banner */}
      {sysStatus?.threshold_level === 'tight' && (
        <div className="flex items-start gap-3 bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-xl px-4 py-3">
          <ShieldAlert size={20} className="text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-700 dark:text-red-300">
              Early Warning System Active
            </p>
            <p className="text-xs text-red-600 dark:text-red-400 mt-0.5">
              Adaptive threshold has tightened to {sysStatus.threshold_value?.toFixed(2) ?? 'N/A'}.
              Unusual fraud patterns detected — increased scrutiny is in effect.
            </p>
          </div>
        </div>
      )}

      {/* Active Alerts */}
      <div className="space-y-3">
        <h2 className="text-base font-semibold text-gray-900 dark:text-white flex items-center gap-2">
          <AlertTriangle size={16} className="text-red-500" />
          Active Alerts
          {!loading && <span className="text-sm font-normal text-gray-400">({active.length})</span>}
        </h2>

        {loading ? (
          <div className="space-y-3">
            {Array(3).fill(0).map((_, i) => <Skeleton key={i} className="h-24" />)}
          </div>
        ) : active.length === 0 ? (
          <div className="card flex flex-col items-center py-12 text-center">
            <BellOff size={32} className="text-gray-300 dark:text-gray-600 mb-3" />
            <p className="text-gray-500 font-medium">No active alerts</p>
            <p className="text-xs text-gray-400 mt-1">New flagged transactions will appear here within 15 seconds.</p>
          </div>
        ) : (
          active.map((alert, i) => (
            <div
              key={alert.id || i}
              className="card border-l-4 border-l-red-500 p-4 space-y-3"
            >
              {/* Alert header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <StatusBadge status={alert.status} />
                    <span className="text-sm font-semibold text-gray-900 dark:text-white">
                      ${alert.amount?.toFixed(2)}
                    </span>
                    <span className="text-xs text-gray-400 font-mono">
                      {alert.transaction_id?.slice(0, 16) ?? alert.id?.slice(0, 16)}...
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    User: <span className="font-mono">{maskCard(alert.user_id)}</span>
                    &nbsp;·&nbsp;
                    {new Date(alert.timestamp).toLocaleString()}
                    &nbsp;·&nbsp;
                    <span className={`font-semibold ${alert.risk_score > 0.65 ? 'text-red-600' : 'text-yellow-600'}`}>
                      Risk: {(alert.risk_score * 100).toFixed(1)}%
                    </span>
                  </p>
                </div>

                <div className="flex gap-2 flex-shrink-0">
                  <button
                    onClick={() => setExpanded(expanded === i ? null : i)}
                    className="btn-secondary text-xs py-1.5"
                  >
                    {expanded === i ? 'Hide' : 'Explain'}
                  </button>
                  <button
                    onClick={() => handleAck(alert.id)}
                    disabled={acking === alert.id}
                    className="btn-primary text-xs py-1.5 disabled:opacity-50 flex items-center gap-1"
                  >
                    {acking === alert.id ? (
                      <>
                        <span className="w-3 h-3 border-2 border-white/50 border-t-white rounded-full animate-spin" />
                        Acknowledging...
                      </>
                    ) : 'Acknowledge'}
                  </button>
                </div>
              </div>

              {/* SHAP breakdown */}
              {expanded === i && alert.top_features && (
                <div className="pt-2 border-t border-gray-100 dark:border-gray-700">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    SHAP Feature Importance — Why flagged?
                  </p>
                  <ShapChart features={alert.top_features} />
                </div>
              )}

              {/* No SHAP available message */}
              {expanded === i && !alert.top_features && (
                <div className="pt-2 border-t border-gray-100 dark:border-gray-700 text-xs text-gray-400">
                  SHAP explanation not available for this transaction.
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Acknowledged Alerts */}
      {acked.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-base font-semibold text-gray-600 dark:text-gray-400 flex items-center gap-2">
            <CheckCircle2 size={16} className="text-green-500" />
            Acknowledged
            <span className="text-sm font-normal text-gray-400">({acked.length})</span>
          </h2>
          <div className="space-y-2 opacity-60">
            {acked.slice(0, 5).map((alert, i) => (
              <div key={alert.id || i} className="card py-3 px-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <CheckCircle2 size={14} className="text-green-500 flex-shrink-0" />
                  <span className="text-sm font-medium">${alert.amount?.toFixed(2)}</span>
                  <StatusBadge status={alert.status} />
                </div>
                <span className="text-xs text-gray-400">
                  {new Date(alert.timestamp).toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}