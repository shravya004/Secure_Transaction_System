export default function RiskGauge({ score }) {
  const pct = Math.min(Math.max(score || 0, 0), 1);
  const rotation = (pct - 0.5) * 180;
  const color = pct < 0.35 ? '#22c55e' : pct < 0.65 ? '#f59e0b' : '#ef4444';
  const label = pct < 0.35 ? 'LOW RISK' : pct < 0.65 ? 'REVIEW' : 'HIGH RISK';
  const bgColor = pct < 0.35 ? '#dcfce7' : pct < 0.65 ? '#fef9c3' : '#fee2e2';

  return (
    <div className="flex flex-col items-center">
      <div
        className="rounded-2xl p-4 transition-colors duration-500"
        style={{ backgroundColor: bgColor }}
      >
        <svg viewBox="0 0 200 110" className="w-56 md:w-64">
          {/* Track */}
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e5e7eb" strokeWidth="16" strokeLinecap="round"/>
          {/* Colored arc segments */}
          <path d="M 20 100 A 80 80 0 0 1 86 22" fill="none" stroke="#22c55e" strokeWidth="16" strokeLinecap="round"/>
          <path d="M 86 22 A 80 80 0 0 1 114 22" fill="none" stroke="#f59e0b" strokeWidth="16" strokeLinecap="round"/>
          <path d="M 114 22 A 80 80 0 0 1 180 100" fill="none" stroke="#ef4444" strokeWidth="16" strokeLinecap="round"/>
          {/* Needle */}
          <line
            x1="100" y1="100" x2="100" y2="28"
            stroke={color} strokeWidth="3" strokeLinecap="round"
            transform={`rotate(${rotation}, 100, 100)`}
            style={{ transition: 'transform 0.6s cubic-bezier(.34,1.56,.64,1)' }}
          />
          <circle cx="100" cy="100" r="6" fill={color}/>
          <circle cx="100" cy="100" r="3" fill="white"/>
          {/* Score */}
          <text x="100" y="88" textAnchor="middle" fontSize="20" fontWeight="bold" fill={color}>
            {(pct * 100).toFixed(1)}%
          </text>
        </svg>
      </div>
      <span className="text-base font-bold mt-2 tracking-widest" style={{ color }}>{label}</span>
    </div>
  );
}