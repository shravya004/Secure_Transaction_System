import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Transaction from './pages/Transaction';
import Ledger from './pages/Ledger';
import Alerts from './pages/Alerts';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard"   element={<Dashboard />} />
          <Route path="transaction" element={<Transaction />} />
          <Route path="ledger"      element={<Ledger />} />
          <Route path="alerts"      element={<Alerts />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}