import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function ShapChart({ features }) {
  if (!features || features.length === 0) return null;
  const sorted = [...features].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
        🔴 Red bars push toward fraud &nbsp;|&nbsp; 🔵 Blue bars push away from fraud
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 10, right: 20 }}>
          <XAxis type="number" domain={[-1, 1]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 11 }}/>
          <YAxis type="category" dataKey="name" width={50} tick={{ fontSize: 12 }}/>
          <Tooltip formatter={(v) => v.toFixed(4)} />
          <ReferenceLine x={0} stroke="#9ca3af" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, i) => (
              <Cell key={i} fill={entry.value > 0 ? '#ef4444' : '#3b82f6'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}