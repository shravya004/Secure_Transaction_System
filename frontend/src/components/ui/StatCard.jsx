export default function StatCard({ title, value, subtitle, icon: Icon, color = 'primary' }) {
  const colorMap = {
    primary: 'text-blue-700 bg-blue-50 dark:bg-blue-900/30 dark:text-blue-300',
    danger:  'text-red-700 bg-red-50 dark:bg-red-900/30 dark:text-red-300',
    success: 'text-green-700 bg-green-50 dark:bg-green-900/30 dark:text-green-300',
    warning: 'text-yellow-700 bg-yellow-50 dark:bg-yellow-900/30 dark:text-yellow-300',
  };

  return (
    <div className="card flex items-start gap-4">
      {Icon && (
        <div className={`p-3 rounded-xl ${colorMap[color]}`}>
          <Icon size={22} />
        </div>
      )}
      <div>
        <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
        <p className="text-2xl font-bold mt-0.5">{value ?? '—'}</p>
        {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}