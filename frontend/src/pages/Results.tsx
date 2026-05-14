import { useLocation, useNavigate, useParams } from 'react-router-dom'
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
    Straightforward: { color: '#10B981', bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)', glow: '0 0 40px rgba(16,185,129,0.2)' },
    Advanced: { color: '#F59E0B', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)', glow: '0 0 40px rgba(245,158,11,0.2)' },
    Complex: { color: '#F43F5E', bg: 'rgba(244,63,94,0.12)', border: 'rgba(244,63,94,0.3)', glow: '0 0 40px rgba(244,63,94,0.2)' },
}

const FACTORS = [
    { key: 'apical_bone', label: 'Apical Bone Availability' },
    { key: 'buccal_wall', label: 'Buccal Wall Thickness' },
    { key: 'ridge_width', label: 'Buccolingual Ridge Width' },
    { key: 'septum_width', label: 'Interradicular Septum' },
    { key: 'periapical_lesion', label: 'Periapical Lesion Status' },
]

export default function Results() {
    const { caseId } = useParams()
    const location = useLocation()
    const navigate = useNavigate()
    const [data, setData] = useState<UploadResponse | null>(location.state?.result || null)
    const [loading, setLoading] = useState(!location.state?.result)
    const [revealed, setRevealed] = useState<number[]>([])

    useEffect(() => {
        if (!data && caseId) {
            getCase(caseId).then(setData).finally(() => setLoading(false))
        }
    }, [caseId])

    useEffect(() => {
        if (!data) return
        FACTORS.forEach((_, i) => {
            setTimeout(() => setRevealed(prev => [...prev, i]), 400 + i * 350)
        })
    }, [data])

    if (loading) return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: "'Plus Jakarta Sans', sans-serif", position: 'relative' }}>
            <Background /><Sidebar />
            <main style={{
                marginLeft: '224px', flex: 1, display: 'flex', alignItems: 'center',
                justifyContent: 'center', position: 'relative', zIndex: 1
            }}>
                <p style={{ color: '#64748B' }}>Loading results...</p>
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

    const { result, yolo } = data
    const cls = CLASS_STYLE[result.classification] || CLASS_STYLE.Complex

    const getMeasurement = (key: string) => {
        const f = result.factors as any
        if (key === 'periapical_lesion') return f.periapical_lesion.lesion_detected ? 'Present' : 'Absent'
        const val = f[key]?.measurement_mm
        return val !== null && val !== undefined ? `${val} mm` : 'N/A'
    }

    const getRisk = (key: string): string => {
        const f = result.factors as any
        if (key === 'periapical_lesion') return f.periapical_lesion.risk
        return f[key]?.risk || 'N/A'
    }

    const card: React.CSSProperties = {
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '16px',
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
                <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'flex-start', marginBottom: '1.5rem'
                }}>
                    <div>
                        <h1 style={{
                            color: '#F1F5F9', fontSize: '1.75rem', fontWeight: 700,
                            margin: 0, letterSpacing: '-0.02em'
                        }}>
                            Results
                        </h1>
                        <p style={{ color: '#64748B', fontSize: '0.8rem', margin: '0.25rem 0 0' }}>
                            Patient: {data.patient_id} · Case: {data.case_id.slice(0, 8)}...
                        </p>
                    </div>
                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <button
                            onClick={() => navigate('/upload')}
                            style={{
                                padding: '0.6rem 1.25rem', borderRadius: '10px',
                                border: '1px solid rgba(99,102,241,0.3)',
                                background: 'rgba(99,102,241,0.08)', color: '#A78BFA',
                                fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer', fontFamily: 'inherit',
                            }}
                        >
                            + New Case
                        </button>
                        <button
                            onClick={() => navigate('/history')}
                            style={{
                                padding: '0.6rem 1.25rem', borderRadius: '10px',
                                border: '1px solid rgba(255,255,255,0.08)',
                                background: 'transparent', color: '#64748B',
                                fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer', fontFamily: 'inherit',
                            }}
                        >
                            History
                        </button>
                    </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem', minWidth: 0 }}>

                    {/* Left column */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: 0 }}>

                        {/* SAC Badge */}
                        <div style={{
                            ...card,
                            padding: '2rem',
                            border: `1px solid ${cls.border}`,
                            boxShadow: cls.glow,
                            textAlign: 'center',
                        }}>
                            <p style={{
                                color: '#64748B', fontSize: '0.7rem', fontWeight: 600,
                                textTransform: 'uppercase', letterSpacing: '0.12em', margin: '0 0 0.75rem'
                            }}>
                                Final Classification
                            </p>
                            <h2 style={{
                                color: cls.color, fontSize: '3rem', fontWeight: 900,
                                margin: 0, letterSpacing: '0.05em',
                                textShadow: `0 0 30px ${cls.color}60`,
                            }}>
                                {result.classification.toUpperCase()}
                            </h2>
                        </div>

                        {/* Factor table */}
                        <div style={{ ...card, overflow: 'hidden' }}>
                            <div style={{
                                display: 'grid', gridTemplateColumns: '1fr 120px 100px',
                                padding: '0.75rem 1.5rem',
                                borderBottom: '1px solid rgba(255,255,255,0.05)',
                            }}>
                                {['Clinical Factor', 'Measurement', 'Risk'].map(h => (
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
                                const measure = getMeasurement(factor.key)
                                const shown = revealed.includes(i)

                                return (
                                    <div
                                        key={factor.key}
                                        style={{
                                            display: 'grid', gridTemplateColumns: '1fr 120px 100px',
                                            padding: '1rem 1.5rem',
                                            borderBottom: '1px solid rgba(255,255,255,0.04)',
                                            alignItems: 'center',
                                            background: i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
                                            opacity: shown ? 1 : 0,
                                            transform: shown ? 'translateY(0)' : 'translateY(8px)',
                                            transition: 'opacity 0.4s ease, transform 0.4s ease',
                                        }}
                                    >
                                        <span style={{ color: '#E2E8F0', fontSize: '0.875rem', fontWeight: 500 }}>
                                            {factor.label}
                                        </span>
                                        <span style={{ color: '#00B4D8', fontSize: '0.875rem', fontFamily: 'monospace' }}>
                                            {shown ? measure : ''}
                                        </span>
                                        <div>
                                            {shown && (
                                                <span style={{
                                                    display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
                                                    padding: '0.25rem 0.75rem', borderRadius: '20px',
                                                    fontSize: '0.72rem', fontWeight: 700,
                                                    color: colors.color, background: colors.bg,
                                                    border: `1px solid ${colors.border}`,
                                                    boxShadow: `0 0 8px ${colors.glow}`,
                                                }}>
                                                    ● {risk}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                )
                            })}
                        </div>

                        {/* Disclaimer */}
                        <p style={{
                            color: '#374151', fontSize: '0.72rem', lineHeight: 1.6,
                            padding: '1rem 1.25rem', borderRadius: '12px',
                            background: 'rgba(255,255,255,0.02)',
                            border: '1px solid rgba(255,255,255,0.04)',
                        }}>
                            {result.disclaimer}
                        </p>
                    </div>

                    {/* Right column */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                        {/* Detection info */}
                        <div style={{ ...card, padding: '1.25rem' }}>
                            <p style={{
                                color: '#475569', fontSize: '0.68rem', fontWeight: 600,
                                textTransform: 'uppercase', letterSpacing: '0.08em', margin: '0 0 1rem'
                            }}>
                                Detection Details
                            </p>
                            {[
                                { label: 'Scanner', value: yolo.scanner },
                                { label: 'Confidence', value: `${(yolo.conf * 100).toFixed(1)}%` },
                                { label: 'Site Type', value: yolo.is_molar ? 'Molar' : 'Anterior' },
                                { label: 'Z Range', value: `${yolo.z_range[0]}–${yolo.z_range[1]}` },
                            ].map(item => (
                                <div key={item.label} style={{
                                    display: 'flex', justifyContent: 'space-between',
                                    alignItems: 'center', padding: '0.5rem 0',
                                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                                }}>
                                    <span style={{ color: '#64748B', fontSize: '0.8rem' }}>{item.label}</span>
                                    <span style={{ color: '#F1F5F9', fontSize: '0.8rem', fontWeight: 500 }}>{item.value}</span>
                                </div>
                            ))}
                        </div>

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
                                borderRadius: '12px', padding: '2rem',
                                textAlign: 'center',
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
                                {result.reasoning.map((line, i) => (
                                    <p key={i} style={{
                                        color: i === result.reasoning.length - 1 ? '#A78BFA' : '#64748B',
                                        fontSize: '0.75rem', margin: 0, lineHeight: 1.5,
                                        fontWeight: i === result.reasoning.length - 1 ? 600 : 400,
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