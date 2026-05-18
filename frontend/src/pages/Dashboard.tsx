import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getAllCases } from '../api/client'
import type { CaseSummary } from '../api/client'
import Sidebar from '../components/Sidebar'
import Background from '../components/Background'

const CLASS_COLORS: Record<string, { color: string; bg: string; border: string }> = {
    Straightforward: { color: '#10B981', bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.3)' },
    Advanced: { color: '#F59E0B', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.3)' },
    Complex: { color: '#F43F5E', bg: 'rgba(244,63,94,0.1)', border: 'rgba(244,63,94,0.3)' },
}

export default function Dashboard() {
    const navigate = useNavigate()
    const [cases, setCases] = useState<CaseSummary[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        getAllCases().then(setCases).finally(() => setLoading(false))
    }, [])

    const total = cases.length
    const straightforward = cases.filter(c => c.classification === 'Straightforward').length
    const advanced = cases.filter(c => c.classification === 'Advanced').length
    const complex = cases.filter(c => c.classification === 'Complex').length
    const recent = cases.slice(0, 5)

    const formatDate = (iso: string) => new Date(iso).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
    })

    const card: React.CSSProperties = {
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '16px',
        boxShadow: '0 0 30px rgba(99,102,241,0.06)',
    }

    return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: "'Plus Jakarta Sans', sans-serif", position: 'relative' }}>
            <Background />
            <Sidebar />
            <main style={{ marginLeft: '224px', flex: 1, padding: '2rem', overflowY: 'auto', position: 'relative', zIndex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
                    <div>
                        <h1 style={{ color: '#F1F5F9', fontSize: '1.75rem', fontWeight: 700, margin: 0, letterSpacing: '-0.02em' }}>Dashboard</h1>
                        <p style={{ color: '#94A3B8', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>Overview of your implant planning cases</p>
                    </div>
                    <button onClick={() => navigate('/upload')} style={{
                        padding: '0.75rem 1.5rem', borderRadius: '12px', border: 'none',
                        background: 'linear-gradient(135deg, #6366F1 0%, #00B4D8 100%)',
                        color: 'white', fontWeight: 700, fontSize: '0.875rem',
                        cursor: 'pointer', fontFamily: 'inherit',
                        boxShadow: '0 0 20px rgba(99,102,241,0.3)', flexShrink: 0,
                    }}>+ New Case</button>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0,1fr))', gap: '1rem', marginBottom: '2rem' }}>
                    {[
                        { label: 'Total Cases', value: total, color: '#6366F1', glow: 'rgba(99,102,241,0.15)' },
                        { label: 'Straightforward', value: straightforward, color: '#10B981', glow: 'rgba(16,185,129,0.15)' },
                        { label: 'Advanced', value: advanced, color: '#F59E0B', glow: 'rgba(245,158,11,0.15)' },
                        { label: 'Complex', value: complex, color: '#F43F5E', glow: 'rgba(244,63,94,0.15)' },
                    ].map(stat => (
                        <div key={stat.label} style={{ ...card, padding: '1.5rem', borderColor: stat.color + '20', boxShadow: `0 0 24px ${stat.glow}` }}>
                            <p style={{ color: '#94A3B8', fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', margin: '0 0 0.75rem' }}>{stat.label}</p>
                            <p style={{ color: stat.color, fontSize: '2.8rem', fontWeight: 800, margin: 0, lineHeight: 1, letterSpacing: '-0.03em' }}>{loading ? '—' : stat.value}</p>
                        </div>
                    ))}
                </div>

                <div style={{ ...card, padding: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
                        <h2 style={{ color: '#F1F5F9', fontSize: '1rem', fontWeight: 600, margin: 0 }}>Recent Cases</h2>
                        <button onClick={() => navigate('/history')} style={{ background: 'none', border: 'none', color: '#6366F1', fontSize: '0.8rem', cursor: 'pointer', fontFamily: 'inherit' }}>View all →</button>
                    </div>

                    {loading ? (
                        <p style={{ color: '#94A3B8', textAlign: 'center', padding: '2rem 0' }}>Loading...</p>
                    ) : recent.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '3rem 0' }}>
                            <p style={{ color: '#94A3B8', marginBottom: '1rem' }}>No cases yet.</p>
                            <button onClick={() => navigate('/upload')} style={{ padding: '0.75rem 1.5rem', borderRadius: '12px', border: 'none', background: 'linear-gradient(135deg, #6366F1 0%, #00B4D8 100%)', color: 'white', fontWeight: 600, fontSize: '0.875rem', cursor: 'pointer', fontFamily: 'inherit' }}>Upload your first scan</button>
                        </div>
                    ) : (
                        <div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 180px 130px 60px', padding: '0.5rem 1rem', marginBottom: '0.25rem' }}>
                                {['Patient ID', 'File', 'Classification', 'Date', ''].map(h => (
                                    <span key={h} style={{ color: '#64748B', fontSize: '0.68rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{h}</span>
                                ))}
                            </div>
                            {recent.map((c, i) => {
                                const cls = CLASS_COLORS[c.classification] || CLASS_COLORS.Complex
                                return (
                                    <div key={c.case_id} onClick={() => navigate(`/history/${c.case_id}`)}
                                        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 180px 130px 60px', padding: '0.875rem 1rem', borderRadius: '10px', alignItems: 'center', cursor: 'pointer', transition: 'background 0.15s', background: i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent' }}
                                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.07)'}
                                        onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent'}
                                    >
                                        <span style={{ color: '#F1F5F9', fontSize: '0.875rem', fontWeight: 500 }}>{c.patient_id}</span>
                                        <span style={{ color: '#94A3B8', fontSize: '0.8rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingRight: '1rem' }}>{c.filename}</span>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.3rem 0.75rem', borderRadius: '20px', fontSize: '0.72rem', fontWeight: 600, color: cls.color, background: cls.bg, border: `1px solid ${cls.border}`, width: 'fit-content' }}>● {c.classification}</span>
                                            {c.is_overridden && (
                                                <span style={{ color: '#F59E0B', fontSize: '0.65rem', fontWeight: 600 }}>✎ Overridden</span>
                                            )}
                                        </div>
                                        <span style={{ color: '#94A3B8', fontSize: '0.8rem' }}>{formatDate(c.created_at)}</span>
                                        <span style={{ color: '#6366F1', fontSize: '0.8rem', textAlign: 'right' }}>View →</span>
                                    </div>
                                )
                            })}
                        </div>
                    )}
                </div>
            </main>
        </div>
    )
}