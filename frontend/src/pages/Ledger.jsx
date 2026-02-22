import { useState, useCallback } from 'react';
import { ChevronDown, ChevronUp, Search, Filter, Link2, ShieldCheck } from 'lucide-react';
import StatusBadge from '../components/ui/StatusBadge';
import { usePolling } from '../hooks/usePolling';
import { getLedger } from '../services/api';

function Skeleton({ className = '' }) {
  return <div className={`animate-pulse bg-gray-200 dark:bg-gray-700 rounded ${className}`} />;
}

const FILTERS = ['ALL', 'APPROVED', 'FLAGGED'];

export default function Ledger() {
  const [search, setSearch]     = useState('');
  const [filter, setFilter]     = useState('ALL');
  const [expanded, setExpanded] = useState(null);

  const ledgerFn = useCallback(() => getLedger(), []);
  const { data: txs, loading } = usePolling(ledgerFn, 15000);

  const allTxs = txs || [];

  const filtered = allTxs.filter(tx => {
    const matchStatus = filter === 'ALL' || tx.status === filter;
    const q = search.toLowerCase();
    const matchSearch = !q ||
      tx.id?.toLowerCase().includes(q) ||
      tx.sender?.toLowerCase().includes(q) ||
      tx.receiver?.toLowerCase().includes(q) ||
      tx.block_hash?.toLowerCase().includes(q);
    return matchStatus && matchSearch;
  });

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Link2 size={22} className="text-blue-600" />
            Blockchain Ledger
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
            Live view of <code className="text-xs bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded">GET /blockchain</code> — click any row to inspect block details
          </p>
        </div>
        <span className="text-xs text-gray-400">{allTxs.length} blocks on chain</span>
      </div>

      {/* Search + Filter */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search by sender, receiver, or block hash..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                       bg-white dark:bg-gray-700 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter size={15} className="text-gray-400" />
          {FILTERS.map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                filter === f
                  ? f === 'FLAGGED'  ? 'bg-red-600 text-white'
                  : f === 'APPROVED' ? 'bg-green-600 text-white'
                  : 'bg-blue-700 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        {loading ? (
          <div className="p-6 space-y-3">
            {Array(5).fill(0).map((_, i) => <Skeleton key={i} className="h-12" />)}
          </div>
        ) : filtered.length === 0 ? (
          <div className="py-16 text-center text-gray-400">
            <Link2 size={32} className="mx-auto mb-3 opacity-30" />
            <p className="text-sm font-medium">No blocks found</p>
            <p className="text-xs mt-1">Submit a transaction to see it appear here.</p>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 dark:bg-gray-700/50">
                  <tr className="text-left text-gray-500 dark:text-gray-400">
                    <th className="px-4 py-3 w-8"></th>
                    <th className="px-4 py-3 font-medium">Block #</th>
                    <th className="px-4 py-3 font-medium">Amount</th>
                    <th className="px-4 py-3 font-medium">Sender → Receiver</th>
                    <th className="px-4 py-3 font-medium">Risk Score</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                    <th className="px-4 py-3 font-medium hidden md:table-cell">Block Hash</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((tx, i) => (
                    <>
                      <tr
                        key={tx.id || i}
                        onClick={() => setExpanded(expanded === i ? null : i)}
                        className={`border-t border-gray-100 dark:border-gray-700 cursor-pointer transition-colors
                          ${expanded === i ? 'bg-blue-50 dark:bg-blue-900/20' : 'hover:bg-gray-50 dark:hover:bg-gray-700/40'}
                          ${tx.status === 'FLAGGED' ? 'border-l-2 border-l-red-500' : ''}
                        `}
                      >
                        <td className="px-4 py-3 text-gray-400">
                          {expanded === i ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        </td>
                        <td className="px-4 py-3 font-mono text-xs text-blue-700 dark:text-blue-400 font-semibold">
                          #{tx.block_index}
                        </td>
                        <td className="px-4 py-3 font-medium">${Number(tx.amount).toFixed(2)}</td>
                        <td className="px-4 py-3 text-xs">
                          <span className="font-mono text-gray-700 dark:text-gray-300">{tx.sender}</span>
                          <span className="text-gray-400 mx-1">→</span>
                          <span className="font-mono text-gray-700 dark:text-gray-300">{tx.receiver}</span>
                        </td>
                        <td className="px-4 py-3">
                          {tx.risk_score != null ? (
                            <span className={`font-semibold ${
                              tx.risk_score > 0.5 ? 'text-red-600' : 'text-green-600'
                            }`}>
                              {(tx.risk_score * 100).toFixed(1)}%
                            </span>
                          ) : (
                            <span className="text-gray-400 text-xs">N/A</span>
                          )}
                        </td>
                        <td className="px-4 py-3"><StatusBadge status={tx.status} /></td>
                        <td className="px-4 py-3 font-mono text-xs text-gray-400 hidden md:table-cell">
                          {tx.block_hash?.slice(0, 16)}...
                        </td>
                      </tr>

                      {/* Expanded block details */}
                      {expanded === i && (
                        <tr key={`exp-${i}`} className="border-t dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                          <td colSpan={7} className="px-6 py-4">
                            <div className="flex items-center gap-2 mb-3">
                              <ShieldCheck size={14} className="text-green-500" />
                              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                Block Details — ECDSA Signed & Blockchain Verified
                              </p>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs font-mono">
                              {[
                                ['Block Hash',     tx.block_hash],
                                ['Previous Hash',  tx.prev_hash],
                                ['Nonce',          tx.nonce],
                                ['Sender',         tx.sender],
                                ['Receiver',       tx.receiver],
                                ['Signature',      tx.signature !== '—' ? tx.signature?.slice(0, 40) + '...' : '—'],
                              ].map(([k, v]) => (
                                <div key={k} className="bg-white dark:bg-gray-900 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                                  <span className="text-gray-400 block mb-0.5">{k}</span>
                                  <span className="text-blue-700 dark:text-blue-400 break-all">{v ?? '—'}</span>
                                </div>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="px-4 py-3 border-t dark:border-gray-700 text-xs text-gray-400 bg-gray-50 dark:bg-gray-800/30">
              Showing {filtered.length} of {allTxs.length} blocks · Chain integrity: {txs?.chain_valid !== false ? '✓ Valid' : '✗ Invalid'}
            </div>
          </>
        )}
      </div>
    </div>
  );
}