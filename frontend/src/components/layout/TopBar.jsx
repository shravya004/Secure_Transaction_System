import { Menu, Sun, Moon } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';

export default function TopBar({ onMenuClick }) {
  const { dark, toggle } = useTheme();

  return (
    <header className="h-14 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-4 md:px-6 flex-shrink-0">
      <button
        onClick={onMenuClick}
        className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
      >
        <Menu size={20} />
      </button>

      <span className="text-sm text-gray-500 dark:text-gray-400 hidden md:block">
        Secure Transaction System
      </span>

      <button
        onClick={toggle}
        className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        aria-label="Toggle dark mode"
      >
        {dark ? <Sun size={20} className="text-yellow-400" /> : <Moon size={20} />}
      </button>
    </header>
  );
}