import { useState } from 'react';
import { predictTransaction } from '../services/api';
import RiskGauge from '../components/ui/RiskGauge';
import StatusBadge from '../components/ui/StatusBadge';
import { Send, RotateCcw, AlertCircle, CheckCircle, ShieldAlert, Lock } from 'lucide-react';

const defaultForm = {
  amount:   '',
  time:     '',
  sender:   '',
  receiver: '',
  v1: 0, v2: 0, v3: 0, v4: 0, v5: 0, v6: 0,
};

export default function Transaction() {
  const [form,    setForm]    = useState(defaultForm);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');

  const set = e => setForm(p => ({ ...p, [e.target.name]: e.target.value }));

  const handleSubmit = async () => {
    if (!form.amount || !form.sender || !form.receiver) {
      setError('Amount, Sender ID, and Receiver ID are required.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await predictTransaction(form);
      setResult(res);
    } catch (e) {
      setError(
        e.response?.data?.detail ||
        'Could not reach backend. Is uvicorn running on port 8000?'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => { setForm(defaultForm); setResult(null); setError(''); };

  return (
    <div className="max-w-4xl mx-auto space-y-6">

      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Submit Transaction</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
          Runs through the full pipeline: ECDSA sign → AI risk score → Policy → Blockchain
        </p>
      </div>

      {/* ── Form ─────────────────────────────────────────────────────── */}
      <div className="card space-y-5">

        {/* Core fields */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
            Transaction Details
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { name: 'amount',   label: 'Amount ($)',     type: 'number', placeholder: '250.00',       required: true },
              { name: 'time',     label: 'Time (seconds)', type: 'number', placeholder: '86400',         required: false },
              { name: 'sender',   label: 'Sender ID',      type: 'text',   placeholder: 'user_alice',    required: true },
              { name: 'receiver', label: 'Receiver ID',    type: 'text',   placeholder: 'merchant_001',  required: true },
            ].map(({ name, label, type, placeholder, required }) => (
              <div key={name}>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                  {label} {required && <span className="text-red-500">*</span>}
                </label>
                <input
                  name={name} type={type} placeholder={placeholder}
                  value={form[name]} onChange={set}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2
                             bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                             focus:ring-2 focus:ring-blue-500 outline-none transition"
                />
              </div>
            ))}
          </div>
        </div>

        {/* V-feature sliders */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
            AI Feature Inputs (V1–V6)
          </p>
          <p className="text-xs text-gray-400 mb-3">
            PCA-anonymised behavioural signals — sent as <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">features[]</code> to the PyTorch risk model.
            Slide right (positive) to simulate suspicious behaviour.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {[1,2,3,4,5,6].map(i => (
              <div key={i}>
                <label className="text-xs text-gray-500 dark:text-gray-400 flex justify-between">
                  <span>V{i}</span>
                  <span className={`font-bold ${form[`v${i}`] > 0 ? 'text-red-500' : form[`v${i}`] < 0 ? 'text-blue-500' : 'text-gray-500'}`}>
                    {Number(form[`v${i}`]).toFixed(1)}
                  </span>
                </label>
                <input
                  type="range" name={`v${i}`} min="-3" max="3" step="0.1"
                  value={form[`v${i}`]} onChange={set}
                  className="w-full accent-blue-600 mt-1"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3 pt-1">
          <button
            onClick={handleSubmit} disabled={loading}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {loading
              ? <><span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" /> Processing...</>
              : <><Send size={16} /> Submit Transaction</>
            }
          </button>
          <button onClick={handleReset} className="btn-secondary flex items-center gap-2">
            <RotateCcw size={16} /> Reset
          </button>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm
                          bg-red-50 dark:bg-red-900/20 rounded-lg px-3 py-2">
            <AlertCircle size={16} className="flex-shrink-0" />
            {error}
          </div>
        )}
      </div>

      {/* ── Result ───────────────────────────────────────────────────── */}
      {result && (
        <div className="card space-y-6">

          <h2 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            {result.status === 'APPROVED'
              ? <CheckCircle  size={20} className="text-green-500" />
              : <ShieldAlert  size={20} className="text-red-500"   />
            }
            Analysis Result
          </h2>

          {/* Raw message banner */}
          <div className={`rounded-lg border px-4 py-3 text-sm font-medium
            ${result.status === 'APPROVED'
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700 text-green-800 dark:text-green-200'
              : 'bg-red-50   dark:bg-red-900/20   border-red-200   dark:border-red-700   text-red-800   dark:text-red-200'
            }`}
          >
            {result.raw_message}
          </div>

          <div className="flex flex-col md:flex-row gap-8 items-center">
            <RiskGauge score={result.risk_score} />

            <div className="grid grid-cols-2 gap-3 flex-1 w-full">
              {[
                { label: 'Status',         value: <StatusBadge status={result.status} /> },
                { label: 'Risk Score',     value: <span className="font-bold text-lg">{(result.risk_score * 100).toFixed(2)}%</span> },
                { label: 'AI Score',       value: <span className="font-semibold">{(result.ai_score * 100).toFixed(2)}%</span> },
                { label: 'ECDSA Signing',  value: <span className="text-green-600 dark:text-green-400 font-semibold flex items-center gap-1"><Lock size={12}/> Verified</span> },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">{label}</p>
                  {value}
                </div>
              ))}

              <div className="col-span-2 bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Transaction ID</p>
                <p className="font-mono text-xs break-all text-blue-700 dark:text-blue-400">{result.transaction_id}</p>
              </div>
            </div>
          </div>

          {/* Pipeline steps */}
          <div className="border-t dark:border-gray-700 pt-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
              Pipeline Completed
            </p>
            <div className="flex flex-wrap gap-2">
              {[
                'SHA-256 Hash',
                'ECDSA Sign',
                'Sig Verify',
                'PyTorch AI Score',
                'Policy Decision',
                result.status === 'APPROVED' ? 'Blockchain Write ✓' : 'Rejected (not written)',
              ].map((step, i) => (
                <span key={i} className={`text-xs px-3 py-1 rounded-full font-medium
                  ${i === 5 && result.status !== 'APPROVED'
                    ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                    : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                  }`}
                >
                  {i + 1}. {step}
                </span>
              ))}
            </div>
          </div>

          <p className="text-xs text-gray-400">
            {result.status === 'APPROVED'
              ? 'This transaction has been committed to the blockchain. View it in the Ledger tab.'
              : 'High-risk transactions are not written to the blockchain.'}
          </p>
        </div>
      )}

    </div>
  );
}