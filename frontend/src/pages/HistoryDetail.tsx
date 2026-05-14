import { useNavigate, useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { getCase } from '../api/client'
import type { UploadResponse } from '../api/client'
import Sidebar from '../components/Sidebar'
import Background from '../components/Background'

const RISK_COLORS = {
    Green: { color: '#10B981', bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.25)', glow: 'rgba(16,185,129,0.2)' },
    Yellow: { color: '#F59E0B', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.25)', glow: 'rgba(245,158,11,0.2)' },
    Red: { color: '#F43F5E', bg: 'rgba(244,63,94,0.1)', border: 'rgba(244,63,94,0.25)', glow: 'rgba(244,63,94,0.2)' },
    'N/A': { color: '#475569', bg: 'rgba(71,85,105,0.1)', border: 'rgba(71,85,105,0.25)', glow: 'transparent' },
}

const CLASS_STYLE = {
    Straightforward: { color: '#10B981', border: 'rgba(16,185,129,0.3)', glow: '0 0 30px rgba(16,185,129,0.15)' },
    Advanced: { color: '#F59E0B', border: 'rgba(245,158,11,0.3)', glow: '0 0 30px rgba(245,158,11,0.15)' },
    Complex: { color: '#F43F5E', border: 'rgba(244,63,94,0.3)', glow: '0 0 30px rgba(244,63,94,0.15)' },
}

const FACTORS = [
    { key: 'apical_bone', label: 'Apical Bone Availability' },
    { key: 'buccal_wall', label: 'Buccal Wall Thickness' },
    { key: 'ridge_width', label: 'Buccolingual Ridge Width' },
    { key: 'septum_width', label: 'Interradicular Septum' },
    { key: 'periapical_lesion', label: 'Periapical Lesion Status' },
]

export default function HistoryDetail() {
    const { caseId } = useParams()
    const navigate = useNavigate()
    const [data, setData] = useState<UploadResponse | null>(null)
    const [loading, setLoading] = useState(true)
    const [notes, setNotes] = useState('')
    const [notesSaved, setNotesSaved] = useState(false)

    useEffect(() => {
        if (caseId) {
            getCase(caseId).then(setData).finally(() => setLoading(false))
        }
    }, [caseId])

    const getMeasurement = (key: string) => {
        if (!data) return ''
        const f = data.result.factors as any
        if (key === 'periapical_lesion') return f.periapical_lesion.lesion_detected ? 'Present' : 'Absent'
        const val = f[key]?.measurement_mm
        return val !== null && val !== undefined ? `${val} mm` : 'N/A'
    }

    const getRisk = (key: string): string => {
        if (!data) return 'N/A'
        const f = data.result.factors as any
        if (key === 'periapical_lesion') return f.periapical_lesion.risk
        return f[key]?.risk || 'N/A'
    }

    const card: React.CSSProperties = {
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '16px',
    }

    if (loading) return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: "'Plus Jakarta Sans', sans-serif", position: 'relative' }}>
            <Background /><Sidebar />
            <main style={{
                marginLeft: '224px', flex: 1, display: 'flex', alignItems: 'center',
                justifyContent: 'center', position: 'relative', zIndex: 1
            }}>
                <p style={{ color: '#64748B' }}>Loading case...</p>
            </main>
        </div>
    )

    if (!data) return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: "'Plus Jakarta Sans', sans-serif", position: 'relative' }}>
            <Background /><Sidebar />
            <main style={{
                marginLeft: '224px', flex: 1, display: 'flex', alignItems: 'center',
                justifyContent: 'center', position: 'relative', zIndex: 1
            }}>
                <p style={{ color: '#F43F5E' }}>Case not found.</p>
            </main>
        </div>
    )

    const cls = CLASS_STYLE[data.result.classification] || CLASS_STYLE.Complex

    return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: "'Plus Jakarta Sans', sans-serif", position: 'relative' }}>
            <Background />
            <Sidebar />

            <main style={{
                marginLeft: '224px', flex: 1, padding: '2rem',
                overflowY: 'auto', position: 'relative', zIndex: 1, minWidth: 0
            }}>

                {/* Header */}
                <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'flex-start', marginBottom: '1.5rem'
                }}>
                    <div>
                        <button
                            onClick={() => navigate('/history')}
                            style={{
                                background: 'none', border: 'none', color: '#6366F1',
                                fontSize: '0.8rem', cursor: 'pointer', fontFamily: 'inherit',
                                padding: 0, marginBottom: '0.5rem', display: 'block'
                            }}
                        >
                            ← Back to History
                        </button>
                        <h1 style={{
                            color: '#F1F5F9', fontSize: '1.75rem', fontWeight: 700,
                            margin: 0, letterSpacing: '-0.02em'
                        }}>
                            Case Detail
                        </h1>
                        <p style={{ color: '#64748B', fontSize: '0.8rem', margin: '0.25rem 0 0' }}>
                            Patient: {data.patient_id} · {caseId?.slice(0, 8)}...
                        </p>
                    </div>
                    <div style={{
                        padding: '0.5rem 1.25rem', borderRadius: '10px',
                        border: `1px solid ${cls.border}`,
                        boxShadow: cls.glow,
                    }}>
                        <span style={{ color: cls.color, fontWeight: 700, fontSize: '1rem' }}>
                            {data.result.classification}
                        </span>
                    </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '1.5rem', minWidth: 0 }}>

                    {/* Left */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: 0 }}>

                        {/* Factor table */}
                        <div style={{ ...card, overflow: 'hidden' }}>
                            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <h2 style={{ color: '#F1F5F9', fontSize: '0.95rem', fontWeight: 600, margin: 0 }}>
                                    Clinical Measurements
                                </h2>
                            </div>
                            <div style={{
                                display: 'grid', gridTemplateColumns: '1fr 120px 100px',
                                padding: '0.6rem 1.5rem',
                                borderBottom: '1px solid rgba(255,255,255,0.04)',
                            }}>
                                {['Factor', 'Value', 'Risk'].map(h => (
                                    <span key={h} style={{
                                        color: '#475569', fontSize: '0.68rem',
                                        fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em'
                                    }}>
                                        {h}
                                    </span>
                                ))}
                            </div>
                            {FACTORS.map((factor, i) => {
                                const risk = getRisk(factor.key) as keyof typeof RISK_COLORS
                                const colors = RISK_COLORS[risk] || RISK_COLORS['N/A']
                                return (
                                    <div key={factor.key} style={{
                                        display: 'grid', gridTemplateColumns: '1fr 120px 100px',
                                        padding: '0.875rem 1.5rem', alignItems: 'center',
                                        borderBottom: '1px solid rgba(255,255,255,0.04)',
                                        background: i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
                                    }}>
                                        <span style={{ color: '#E2E8F0', fontSize: '0.875rem', fontWeight: 500 }}>
                                            {factor.label}
                                        </span>
                                        <span style={{ color: '#00B4D8', fontSize: '0.875rem', fontFamily: 'monospace' }}>
                                            {getMeasurement(factor.key)}
                                        </span>
                                        <span style={{
                                            display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
                                            padding: '0.25rem 0.75rem', borderRadius: '20px',
                                            fontSize: '0.72rem', fontWeight: 700,
                                            color: colors.color, background: colors.bg,
                                            border: `1px solid ${colors.border}`,
                                            boxShadow: `0 0 6px ${colors.glow}`,
                                            width: 'fit-content',
                                        }}>
                                            ● {risk}
                                        </span>
                                    </div>
                                )
                            })}
                        </div>

                        {/* Clinician Notes */}
                        <div style={{ ...card, padding: '1.25rem' }}>
                            <h2 style={{
                                color: '#F1F5F9', fontSize: '0.95rem', fontWeight: 600,
                                margin: '0 0 1rem'
                            }}>
                                Clinician Notes
                            </h2>
                            <textarea
                                value={notes}
                                onChange={e => { setNotes(e.target.value); setNotesSaved(false) }}
                                placeholder="Add clinical notes, observations, or treatment plan details..."
                                rows={4}
                                style={{
                                    width: '100%', padding: '0.875rem 1rem', borderRadius: '12px',
                                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(99,102,241,0.2)',
                                    color: '#F1F5F9', fontSize: '0.875rem', outline: 'none',
                                    boxSizing: 'border-box', colorScheme: 'dark', fontFamily: 'inherit',
                                    resize: 'vertical', lineHeight: 1.6,
                                }}
                                onFocus={e => e.target.style.borderColor = 'rgba(99,102,241,0.5)'}
                                onBlur={e => e.target.style.borderColor = 'rgba(99,102,241,0.2)'}
                            />
                            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.75rem' }}>
                                <button
                                    onClick={() => setNotesSaved(true)}
                                    style={{
                                        padding: '0.6rem 1.25rem', borderRadius: '10px',
                                        background: notesSaved ? 'rgba(16,185,129,0.15)' : 'rgba(99,102,241,0.15)',
                                        color: notesSaved ? '#10B981' : '#A78BFA',
                                        fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer', fontFamily: 'inherit',
                                        border: `1px solid ${notesSaved ? 'rgba(16,185,129,0.3)' : 'rgba(99,102,241,0.3)'}`,
                                    }}
                                >
                                    {notesSaved ? '✓ Saved' : 'Save Notes'}
                                </button>
                            </div>
                        </div>

                        {/* Disclaimer */}
                        <p style={{
                            color: '#374151', fontSize: '0.72rem', lineHeight: 1.6,
                            padding: '1rem 1.25rem', borderRadius: '12px',
                            background: 'rgba(255,255,255,0.02)',
                            border: '1px solid rgba(255,255,255,0.04)',
                        }}>
                            {data.result.disclaimer}
                        </p>
                    </div>

                    {/* Right */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                        {/* MPR Viewer placeholder */}
                        <div style={{
                            ...card, padding: '1.25rem',
                            border: '1px solid rgba(99,102,241,0.15)',
                        }}>
                            <p style={{
                                color: '#475569', fontSize: '0.68rem', fontWeight: 600,
                                textTransform: 'uppercase', letterSpacing: '0.08em', margin: '0 0 1rem'
                            }}>
                                MPR Viewer
                            </p>
                            <div style={{
                                background: 'rgba(99,102,241,0.05)',
                                border: '1px dashed rgba(99,102,241,0.2)',
                                borderRadius: '12px', padding: '2rem', textAlign: 'center',
                            }}>
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                                    stroke="#6366F1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                                    style={{ marginBottom: '0.75rem' }}>
                                    <rect x="3" y="3" width="18" height="18" rx="2" />
                                    <circle cx="12" cy="12" r="3" />
                                    <line x1="12" y1="3" x2="12" y2="9" />
                                    <line x1="12" y1="15" x2="12" y2="21" />
                                    <line x1="3" y1="12" x2="9" y2="12" />
                                    <line x1="15" y1="12" x2="21" y2="12" />
                                </svg>
                                <p style={{ color: '#6366F1', fontSize: '0.8rem', fontWeight: 600, margin: '0 0 0.25rem' }}>
                                    Interactive MPR Viewer
                                </p>
                                <p style={{ color: '#374151', fontSize: '0.72rem', margin: 0 }}>
                                    Available after database migration
                                </p>
                            </div>
                        </div>

                        {/* Reasoning */}
                        <div style={{ ...card, padding: '1.25rem' }}>
                            <p style={{
                                color: '#475569', fontSize: '0.68rem', fontWeight: 600,
                                textTransform: 'uppercase', letterSpacing: '0.08em', margin: '0 0 1rem'
                            }}>
                                Reasoning Chain
                            </p>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                {data.result.reasoning.map((line, i) => (
                                    <p key={i} style={{
                                        color: i === data.result.reasoning.length - 1 ? '#A78BFA' : '#64748B',
                                        fontSize: '0.75rem', margin: 0, lineHeight: 1.5,
                                        fontWeight: i === data.result.reasoning.length - 1 ? 600 : 400,
                                    }}>
                                        {line}
                                    </p>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    )
}