import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getAllCases } from '../api/client'
import type { CaseSummary } from '../api/client'
import Sidebar from '../components/Sidebar'
import Background from '../components/Background'

export default function Review() {
    const navigate = useNavigate()
    const [cases, setCases] = useState<CaseSummary[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        getAllCases().then(setCases).finally(() => setLoading(false))
    }, [])

    // For review we show only Complex and Advanced cases
    const reviewCases = cases.filter(c =>
        c.classification === 'Complex' || c.classification === 'Advanced'
    )

    const formatDate = (iso: string) => new Date(iso).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
    })

    const card: React.CSSProperties = {
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '16px',
    }

    const CLASS_COLORS: Record<string, { color: string; bg: string; border: string }> = {
        Advanced: { color: '#F59E0B', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.25)' },
        Complex: { color: '#F43F5E', bg: 'rgba(244,63,94,0.1)', border: 'rgba(244,63,94,0.25)' },
    }

    return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: "'Plus Jakarta Sans', sans-serif", position: 'relative' }}>
            <Background />
            <Sidebar />

            <main style={{
                marginLeft: '224px', flex: 1, padding: '2rem',
                overflowY: 'auto', position: 'relative', zIndex: 1, minWidth: 0
            }}>

                {/* Header */}
                <div style={{ marginBottom: '2rem' }}>
                    <h1 style={{
                        color: '#F1F5F9', fontSize: '1.75rem', fontWeight: 700,
                        margin: 0, letterSpacing: '-0.02em'
                    }}>
                        Review
                    </h1>
                    <p style={{ color: '#64748B', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
                        Cases requiring clinical attention — Advanced and Complex only
                    </p>
                </div>

                {/* Stats */}
                <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))',
                    gap: '1rem', marginBottom: '2rem'
                }}>
                    {[
                        {
                            label: 'Pending Review', value: reviewCases.length,
                            color: '#F59E0B', glow: 'rgba(245,158,11,0.15)'
                        },
                        {
                            label: 'Advanced',
                            value: cases.filter(c => c.classification === 'Advanced').length,
                            color: '#F59E0B', glow: 'rgba(245,158,11,0.1)'
                        },
                        {
                            label: 'Complex',
                            value: cases.filter(c => c.classification === 'Complex').length,
                            color: '#F43F5E', glow: 'rgba(244,63,94,0.15)'
                        },
                    ].map(stat => (
                        <div key={stat.label} style={{
                            ...card, padding: '1.25rem',
                            borderColor: stat.color + '22',
                            boxShadow: `0 0 20px ${stat.glow}`,
                        }}>
                            <p style={{
                                color: '#64748B', fontSize: '0.7rem', fontWeight: 600,
                                textTransform: 'uppercase', letterSpacing: '0.1em', margin: '0 0 0.5rem'
                            }}>
                                {stat.label}
                            </p>
                            <p style={{
                                color: stat.color, fontSize: '2rem', fontWeight: 800,
                                margin: 0, letterSpacing: '-0.03em'
                            }}>
                                {loading ? '—' : stat.value}
                            </p>
                        </div>
                    ))}
                </div>

                {/* Cases list */}
                <div style={card}>
                    <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <h2 style={{ color: '#F1F5F9', fontSize: '0.95rem', fontWeight: 600, margin: 0 }}>
                            Cases Requiring Review
                        </h2>
                    </div>

                    {loading ? (
                        <div style={{ padding: '3rem', textAlign: 'center', color: '#64748B' }}>Loading...</div>
                    ) : reviewCases.length === 0 ? (
                        <div style={{ padding: '4rem', textAlign: 'center' }}>
                            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>✓</div>
                            <p style={{ color: '#10B981', fontWeight: 600, marginBottom: '0.5rem' }}>
                                No cases require review
                            </p>
                            <p style={{ color: '#64748B', fontSize: '0.875rem' }}>
                                All analyzed cases are Straightforward
                            </p>
                        </div>
                    ) : (
                        reviewCases.map((c, i) => {
                            const cls = CLASS_COLORS[c.classification]
                            return (
                                <div
                                    key={c.case_id}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: '1rem',
                                        padding: '1.25rem 1.5rem',
                                        borderBottom: '1px solid rgba(255,255,255,0.04)',
                                        background: i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
                                        cursor: 'pointer', transition: 'background 0.15s',
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.07)'}
                                    onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent'}
                                    onClick={() => navigate(`/history/${c.case_id}`)}
                                >
                                    {/* Risk indicator */}
                                    <div style={{
                                        width: '4px', height: '40px', borderRadius: '2px', flexShrink: 0,
                                        background: cls?.color,
                                        boxShadow: `0 0 8px ${cls?.color}`,
                                    }} />

                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
                                            <span style={{ color: '#F1F5F9', fontSize: '0.875rem', fontWeight: 600 }}>
                                                {c.patient_id}
                                            </span>
                                            <span style={{
                                                display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
                                                padding: '0.2rem 0.6rem', borderRadius: '20px', fontSize: '0.7rem',
                                                fontWeight: 700, color: cls?.color, background: cls?.bg,
                                                border: `1px solid ${cls?.border}`,
                                            }}>
                                                ● {c.classification}
                                            </span>
                                        </div>
                                        <p style={{
                                            color: '#64748B', fontSize: '0.78rem', margin: 0,
                                            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
                                        }}>
                                            {c.filename} · {formatDate(c.created_at)}
                                        </p>
                                    </div>

                                    <button
                                        onClick={e => { e.stopPropagation(); navigate(`/history/${c.case_id}`) }}
                                        style={{
                                            padding: '0.5rem 1rem', borderRadius: '8px', border: 'none',
                                            background: 'rgba(99,102,241,0.12)', color: '#A78BFA',
                                            fontWeight: 600, fontSize: '0.78rem', cursor: 'pointer',
                                            fontFamily: 'inherit', flexShrink: 0,
                                            outline: '1px solid rgba(99,102,241,0.25)',
                                        }}
                                    >
                                        Review →
                                    </button>
                                </div>
                            )
                        })
                    )}
                </div>
            </main>
        </div>
    )
}