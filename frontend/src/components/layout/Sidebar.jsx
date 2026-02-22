import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Send, Link2, Bell, X, ShieldAlert } from 'lucide-react';

const links = [
  { to: '/dashboard',   icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/transaction', icon: Send,             label: 'New Transaction' },
  { to: '/ledger',      icon: Link2,            label: 'Ledger' },
  { to: '/alerts',      icon: Bell,             label: 'Alerts' },
];

export default function Sidebar({ open, onClose }) {
  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div className="fixed inset-0 bg-black/50 z-20 md:hidden" onClick={onClose} />
      )}

      <aside className={`
        fixed md:static inset-y-0 left-0 z-30 w-64
        bg-[#0f172a] text-white flex flex-col
        transform transition-transform duration-200
        ${open ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
      `}>
        {/* Logo */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
          <div className="flex items-center gap-2">
            <ShieldAlert size={22} className="text-blue-400" />
            <span className="text-lg font-bold tracking-tight">FRAUD<span className="text-blue-400">-X</span></span>
          </div>
          <button onClick={onClose} className="md:hidden text-gray-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-4 space-y-1">
          {links.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm transition-colors
                ${isActive
                  ? 'bg-blue-600 text-white font-semibold'
                  : 'text-gray-400 hover:bg-white/10 hover:text-white'
                }`
              }
            >
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 text-xs text-gray-600 border-t border-white/10">
          FRAUD-X Framework v1.0 · IEEE Access 2025
        </div>
      </aside>
    </>
  );
}