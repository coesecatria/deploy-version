import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Download, Search, X, Calendar, Database } from 'lucide-react';
import DatePicker from '../components/DatePicker';
import { BRANCHES } from '../utils/branchColors';

// ── PDF Export Modal ──────────────────────────────────────────────────────────
function ExportModal({ currentDate, currentBranch, onClose }) {
    const baseUrl = import.meta.env.VITE_API_URL || '';

    const download = (dateParam) => {
        const params = new URLSearchParams();
        if (dateParam) params.set('date', dateParam);
        if (currentBranch) params.set('branch', currentBranch);
        const url = `${baseUrl}/api/attendance-report/pdf?${params.toString()}`;
        window.open(url, '_blank');
        onClose();
    };

    return (
        <div
            id="pdf-export-modal-overlay"
            style={{
                position: 'fixed', inset: 0, zIndex: 1000,
                background: 'rgba(0,0,0,0.45)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
            onClick={(e) => e.target.id === 'pdf-export-modal-overlay' && onClose()}
        >
            <div style={{
                background: 'var(--card-bg, #fff)',
                border: '1px solid var(--border-color, #e5e7eb)',
                borderRadius: 16,
                padding: '28px 32px',
                width: 380,
                boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
                animation: 'fadeInUp 0.2s ease',
            }}>
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <h3 style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)' }}>Export PDF Report</h3>
                    <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 4, borderRadius: 6 }}>
                        <X size={18} />
                    </button>
                </div>
                <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>
                    Choose the date range for your attendance report.
                    {currentBranch && <> Branch filter <strong>{currentBranch}</strong> will be applied.</>}
                </p>

                {/* Options */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <button
                        id="pdf-export-today"
                        onClick={() => download(currentDate)}
                        style={{
                            display: 'flex', alignItems: 'center', gap: 14,
                            padding: '14px 18px', borderRadius: 10, border: '1.5px solid #0d9488',
                            background: 'rgba(13,148,136,0.06)', cursor: 'pointer',
                            transition: 'all 0.15s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(13,148,136,0.14)'}
                        onMouseLeave={e => e.currentTarget.style.background = 'rgba(13,148,136,0.06)'}
                    >
                        <div style={{ width: 38, height: 38, borderRadius: 8, background: '#ccfbf1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Calendar size={18} color="#0d9488" />
                        </div>
                        <div style={{ textAlign: 'left' }}>
                            <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Today Only</div>
                            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{currentDate}</div>
                        </div>
                    </button>

                    <button
                        id="pdf-export-all"
                        onClick={() => download(null)}
                        style={{
                            display: 'flex', alignItems: 'center', gap: 14,
                            padding: '14px 18px', borderRadius: 10, border: '1.5px solid #6366f1',
                            background: 'rgba(99,102,241,0.06)', cursor: 'pointer',
                            transition: 'all 0.15s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.14)'}
                        onMouseLeave={e => e.currentTarget.style.background = 'rgba(99,102,241,0.06)'}
                    >
                        <div style={{ width: 38, height: 38, borderRadius: 8, background: '#e0e7ff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Database size={18} color="#6366f1" />
                        </div>
                        <div style={{ textAlign: 'left' }}>
                            <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>All Records</div>
                            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Complete attendance history</div>
                        </div>
                    </button>
                </div>
            </div>
        </div>
    );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function AttendancePage() {
    const getLocalDate = () => {
        const d = new Date();
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    };

    const [records, setRecords] = useState([]);
    const [loading, setLoading] = useState(true);
    const [date, setDate] = useState(getLocalDate());
    const [branch, setBranch] = useState('');
    const [rollSearch, setRollSearch] = useState('');
    const [showExportModal, setShowExportModal] = useState(false);

    const fetchRecords = async () => {
        setLoading(true);
        try {
            const params = { limit: 200 };
            if (date) params.date = date;
            if (branch) params.branch = branch;
            if (rollSearch) params.roll_no = rollSearch;
            const data = await api.getAttendanceReport(params);
            setRecords(data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchRecords();
        const interval = setInterval(fetchRecords, 15000);
        return () => clearInterval(interval);
    }, [date, branch, rollSearch]);

    return (
        <>
            {showExportModal && (
                <ExportModal
                    currentDate={date}
                    currentBranch={branch}
                    onClose={() => setShowExportModal(false)}
                />
            )}

            <div className="page-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                    <h2>Attendance Logs</h2>
                    <p>View and export attendance records</p>
                </div>
                <button
                    id="export-pdf-btn"
                    className="btn btn-success"
                    onClick={() => setShowExportModal(true)}
                >
                    <Download size={16} /> Export PDF
                </button>
            </div>

            <div className="page-body">
                {/* Filters */}
                <div className="filter-bar fade-in" style={{ position: 'relative', zIndex: 10 }}>
                    <DatePicker value={date} onChange={setDate} align="right" />

                    <select
                        className="select"
                        value={branch}
                        onChange={(e) => setBranch(e.target.value)}
                        style={{ maxWidth: 160 }}
                    >
                        <option value="">All Branches</option>
                        {BRANCHES.map(b => (
                            <option key={b} value={b}>{b}</option>
                        ))}
                    </select>

                    <div style={{ position: 'relative' }}>
                        <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                        <input
                            className="input"
                            placeholder="Filter by Roll No..."
                            value={rollSearch}
                            onChange={(e) => setRollSearch(e.target.value.toUpperCase())}
                            style={{ paddingLeft: 36, maxWidth: 220 }}
                        />
                    </div>

                    <span style={{ fontSize: 13, color: 'var(--text-muted)', marginLeft: 'auto' }}>
                        {records.length} records
                    </span>
                </div>

                {/* Table */}
                {loading ? (
                    <div className="loading-center"><div className="spinner" /></div>
                ) : (
                    <div className="card fade-in" style={{ padding: 0, overflow: 'visible', position: 'relative', zIndex: 1 }}>
                        <div className="table-wrapper" style={{ overflowX: 'auto', overflowY: 'visible' }}>
                            <table>
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Roll No</th>
                                        <th>Name</th>
                                        <th>Branch</th>
                                        <th>Date</th>
                                        <th>Login</th>
                                        <th>Logout</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {records.length === 0 ? (
                                        <tr>
                                            <td colSpan={7} style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
                                                No attendance records found
                                            </td>
                                        </tr>
                                    ) : (
                                        records.map((r, i) => (
                                            <tr key={i}>
                                                <td style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                                                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{r.roll_no}</td>
                                                <td>{r.name}</td>
                                                <td><span className="badge branch">{r.branch}</span></td>
                                                <td>{r.date}</td>
                                                <td>
                                                    <div style={{ fontFamily: 'monospace', fontWeight: 600 }}>{r.login_time || '--:--:--'}</div>
                                                    {r.login_status && (
                                                        <span className="badge" style={{ backgroundColor: r.login_status === 'On Time' ? 'rgba(16,185,129,0.1)' : 'rgba(245,158,11,0.1)', color: r.login_status === 'On Time' ? '#10b981' : '#f59e0b', fontSize: 10 }}>
                                                            {r.login_status}
                                                        </span>
                                                    )}
                                                </td>
                                                <td>
                                                    <div style={{ fontFamily: 'monospace', fontWeight: 600 }}>{r.logout_time || '--:--:--'}</div>
                                                    {r.logout_status && (
                                                        <span className="badge" style={{ backgroundColor: r.logout_status === 'Logged Out' ? 'rgba(59,130,246,0.1)' : 'rgba(239,68,68,0.1)', color: r.logout_status === 'Logged Out' ? '#3b82f6' : '#ef4444', fontSize: 10 }}>
                                                            {r.logout_status}
                                                        </span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}
