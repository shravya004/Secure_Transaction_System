import { useState } from 'react';
import { predictTransaction } from '../services/api';
import RiskGauge from '../components/ui/RiskGauge';
import StatusBadge from '../components/ui/StatusBadge';
import ShapChart from '../components/ui/ShapChart';
import { Send, RotateCcw, AlertCircle, CheckCircle, ShieldAlert } from 'lucide-react';

const defaultForm = {
  amount: '',
  time: '',
  transaction_type: 'purchase',
  user_id: '',
  receiver: '',
  v1: 0, v2: 0, v3: 0, v4: 0, v5: 0, v6: 0,
};

export default function Transaction() {
  const [form, setForm]       = useState(defaultForm);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  const handleChange = e =>
    setForm(p => ({ ...p, [e.target.name]: e.target.value }));

  const handleSubmit = async () => {
    if (!form.amount || !form.user_id) {
      setError('Amount and Sender ID are required.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await predictTransaction(form);
      setResult(res);
    } catch (e) {
      setError(e.response?.data?.detail || 'Transaction failed. Is the backend running on port 8000?');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm(defaultForm);
    setResult(null);
    setError('');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
        Submit Transaction for Analysis
      </h1>

      {/* Form Card */}
      <div className="card">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
          Transaction Details
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

          {/* Core fields */}
          {[
            { name: 'amount',    label: 'Amount ($)',    type: 'number', placeholder: '0.00' },
            { name: 'user_id',   label: 'Sender ID',     type: 'text',   placeholder: 'user_001' },
            { name: 'receiver',  label: 'Receiver ID',   type: 'text',   placeholder: 'merchant_001' },
            { name: 'time',      label: 'Time (seconds)',type: 'number', placeholder: '3600' },
          ].map(({ name, label, type, placeholder }) => (
            <div key={name}>
              <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">{label}</label>
              <input
                name={name} type={type} placeholder={placeholder}
                value={form[name]} onChange={handleChange}
                className="w-full border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2
                           bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                           focus:ring-2 focus:ring-blue-500 outline-none transition"
              />
            </div>
          ))}

          {/* Transaction type */}
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Transaction Type</label>
            <select
              name="transaction_type" value={form.transaction_type} onChange={handleChange}
              className="w-full border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-blue-500 outline-none"
            >
              {['purchase', 'transfer', 'withdrawal', 'refund'].map(t => (
                <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
              ))}
            </select>
          </div>

          {/* V-feature sliders */}
          <div className="col-span-full">
            <p className="text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
              AI Feature Inputs (V1–V6)
              <span className="ml-2 text-xs text-gray-400 font-normal">
                — PCA-anonymized behavioural signals sent as features[] to the AI model
              </span>
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-2">
              {[1, 2, 3, 4, 5, 6].map(i => (
                <div key={i}>
                  <label className="text-xs text-gray-500 dark:text-gray-400">
                    V{i}: <span className="font-semibold text-gray-700 dark:text-gray-200">{form[`v${i}`]}</span>
                  </label>
                  <input
                    type="range" name={`v${i}`} min="-3" max="3" step="0.1"
                    value={form[`v${i}`]} onChange={handleChange}
                    className="w-full accent-blue-600"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="col-span-full flex gap-3 items-center flex-wrap">
            <button
              onClick={handleSubmit} disabled={loading}
              className="btn-primary flex items-center gap-2 disabled:opacity-50"
            >
              <Send size={16} />
              {loading ? 'Processing...' : 'Submit Transaction'}
            </button>
            <button onClick={handleReset} className="btn-secondary flex items-center gap-2">
              <RotateCcw size={16} />
              Reset
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="col-span-full flex items-center gap-2 text-red-600 dark:text-red-400
                            text-sm bg-red-50 dark:bg-red-900/20 rounded-lg px-3 py-2">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Result Card */}
      {result && (
        <div className="card space-y-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            {result.status === 'APPROVED'
              ? <CheckCircle size={20} className="text-green-500" />
              : <ShieldAlert size={20} className="text-red-500" />
            }
            Analysis Result
          </h2>

          {/* Raw message from backend */}
          <div className={`px-4 py-3 rounded-lg text-sm font-medium border
            ${result.status === 'APPROVED'
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700 text-green-800 dark:text-green-200'
              : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700 text-red-800 dark:text-red-200'
            }`}>
            {result.raw_message}
          </div>

          <div className="flex flex-col md:flex-row gap-8 items-center">
            <RiskGauge score={result.risk_score} />

            <div className="grid grid-cols-2 gap-4 flex-1">
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Status</p>
                <StatusBadge status={result.status} />
              </div>
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Risk Score</p>
                <p className="font-bold text-lg">{(result.risk_score * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">AI Score</p>
                <p className="font-semibold">{(result.ai_score * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Crypto Verified</p>
                <p className="font-semibold text-green-600">✓ ECDSA Valid</p>
              </div>
              <div className="col-span-2 bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Transaction ID</p>
                <p className="font-mono text-xs break-all text-blue-700 dark:text-blue-400">
                  {result.transaction_id}
                </p>
              </div>
            </div>
          </div>

          {/* SHAP — only show if backend provides it */}
          {result.top_features && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">
                Why was this flagged?
              </h3>
              <ShapChart features={result.top_features} />
            </div>
          )}

          {/* Note when SHAP not available */}
          {!result.top_features && result.status === 'FLAGGED' && (
            <div className="text-xs text-gray-400 border-t dark:border-gray-700 pt-4">
              SHAP feature explanation not yet available — will show here once the backend /explain endpoint is added.
            </div>
          )}
        </div>
      )}
    </div>
  );
}