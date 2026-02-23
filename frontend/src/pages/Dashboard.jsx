import { useCallback } from 'react';
import { Activity, AlertTriangle, CheckCircle, TrendingUp, RefreshCw } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, CartesianGrid,
} from 'recharts';
import StatCard from '../components/ui/StatCard';
import StatusBadge from '../components/ui/StatusBadge';
import { usePolling } from '../hooks/usePolling';
import { getDashboardStats, getLedger } from '../services/api';

const PIE_COLORS = { APPROVED: '#22c55e', FLAGGED: '#ef4444', REVIEW: '#f59e0b' };

function Skeleton({ className = '' }) {
  return <div className={`animate-pulse bg-gray-200 dark:bg-gray-700 rounded ${className}`} />;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg shadow-lg px-3 py-2 text-sm">
      <p className="font-semibold text-gray-700 dark:text-gray-200 mb-1">{label}:00</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <span className="font-bold">{p.value}</span>
        </p>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const statsFn  = useCallback(() => getDashboardStats(), []);
  const ledgerFn = useCallback(() => getLedger(), []);

  const { data: stats,  loading: statsLoading,  lastUpdated, refetch } = usePolling(statsFn,  30000);
  const { data: ledger, loading: ledgerLoading }                        = usePolling(ledgerFn, 15000);

  const recent  = ledger?.slice(0, 10) || [];
  const pieData = stats?.status_counts
    ? Object.entries(stats.status_counts).map(([name, value]) => ({ name, value }))
    : [];

  const trendData = stats?.hourly_trend || Array.from({ length: 12 }, (_, i) => ({
    hour: `${(i * 2).toString().padStart(2, '0')}`,
    flagged: Math.floor(Math.random() * 5),
    total: 20 + Math.floor(Math.random() * 30),
  }));

  return (
    <div className="space-y-6">

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">FRAUD-X live transaction overview</p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-gray-400">Updated {lastUpdated.toLocaleTimeString()}</span>
          )}
          <button onClick={refetch} className="btn-secondary flex items-center gap-1 text-sm py-1.5">
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {statsLoading ? (
          Array(4).fill(0).map((_, i) => <Skeleton key={i} className="h-28" />)
        ) : (
          <>
            <StatCard icon={Activity} color="primary" title="Total Today"
              value={stats?.total_today?.toLocaleString() ?? '—'} subtitle="Transactions processed" />
            <StatCard icon={AlertTriangle} color="danger" title="Flagged"
              value={stats?.flagged_count?.toLocaleString() ?? '—'} subtitle="Require review" />
            <StatCard icon={TrendingUp} color="warning" title="Fraud Rate"
              value={stats?.fraud_rate != null ? `${(stats.fraud_rate * 100).toFixed(2)}%` : '—'}
              subtitle="Of total transactions" />
            <StatCard icon={CheckCircle} color="success" title="Avg Risk Score"
              value={stats?.avg_risk != null ? `${(stats.avg_risk * 100).toFixed(1)}%` : '—'}
              subtitle="Across all transactions" />
          </>
        )}
      </div>

      {stats?.threshold_level === 'tight' && (
        <div className="flex items-center gap-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 rounded-xl px-4 py-3">
          <AlertTriangle size={18} className="text-red-600 flex-shrink-0" />
          <p className="text-sm text-red-700 dark:text-red-300 font-medium">
            Early Warning Active — Adaptive threshold has tightened. Unusual fraud spike detected.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white">Fraud Detections (Last 24h)</h2>
            <div className="flex items-center gap-4 text-xs text-gray-500">
              <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-500 inline-block rounded" /> Flagged</span>
              <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500 inline-block rounded" /> Total</span>
            </div>
          </div>
          {statsLoading ? <Skeleton className="h-52" /> : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trendData} margin={{ left: -10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} tickFormatter={v => `${v}h`} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="flagged" name="Flagged" stroke="#ef4444" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="total"   name="Total"   stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="card">
          <h2 className="text-base font-semibold mb-4 text-gray-900 dark:text-white">Status Breakdown</h2>
          {statsLoading ? <Skeleton className="h-52" /> : pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name"
                  cx="50%" cy="50%" outerRadius={75} innerRadius={40}
                  label={({ percent }) => `${(percent * 100).toFixed(0)}%`} labelLine={false}>
                  {pieData.map(e => (
                    <Cell key={e.name} fill={PIE_COLORS[e.name] || '#888'} />
                  ))}
                </Pie>
                <Legend iconType="circle" iconSize={10} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-52 text-gray-400 text-sm">No data yet</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white">Recent Activity</h2>
          <span className="text-xs text-gray-400">Auto-refreshes every 15s</span>
        </div>
        {ledgerLoading ? (
          <div className="space-y-2">{Array(5).fill(0).map((_, i) => <Skeleton key={i} className="h-10" />)}</div>
        ) : recent.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-8">No transactions yet. Submit one from the Transaction page.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  {['TX ID', 'Amount', 'Risk Score', 'Status', 'Time'].map(h => (
                    <th key={h} className="pb-3 pr-4 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recent.map((tx, i) => (
                  <tr key={tx.id || i} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/40 transition-colors">
                    <td className="py-3 pr-4 font-mono text-xs text-blue-700 dark:text-blue-400">{tx.id?.slice(0, 12)}...</td>
                    <td className="py-3 pr-4 font-medium">${tx.amount?.toFixed(2)}</td>
                    <td className="py-3 pr-4">
                      <span className={`font-semibold ${tx.risk_score > 0.65 ? 'text-red-600' : tx.risk_score > 0.35 ? 'text-yellow-600' : 'text-green-600'}`}>
                        {(tx.risk_score * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 pr-4"><StatusBadge status={tx.status} /></td>
                    <td className="py-3 pr-4 text-gray-400 text-xs">{new Date(tx.timestamp).toLocaleTimeString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}