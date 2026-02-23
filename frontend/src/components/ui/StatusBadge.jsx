const styles = {
  APPROVED: 'bg-green-50 text-green-700 border border-green-500',
  FLAGGED:  'bg-red-50 text-red-700 border border-red-500',
  REVIEW:   'bg-yellow-50 text-yellow-700 border border-yellow-500',
};

export default function StatusBadge({ status }) {
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-semibold ${styles[status] || 'bg-gray-100 text-gray-600'}`}>
      {status}
    </span>
  );
}