import { useEffect, useState } from 'react'
import { getVolumeInfo, getSliceUrl } from '../api/client'
import type { VolumeInfo } from '../api/client'

interface MPRViewerProps {
    caseId: string
    layout?: 'horizontal' | 'vertical'
}

function SlicePanel({
    caseId, view, index, max, label, onChange,
}: {
    caseId: string
    view: 'axial' | 'coronal' | 'sagittal'
    index: number
    max: number
    label: string
    onChange: (v: number) => void
}) {
    const [imgUrl, setImgUrl] = useState<string | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        setLoading(true)
        const url = getSliceUrl(caseId, view, index)
        const img = new window.Image()
        img.onload = () => { setImgUrl(url); setLoading(false) }
        img.onerror = () => setLoading(false)
        img.src = url
    }, [caseId, view, index])

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{
                    color: '#94A3B8', fontSize: '0.68rem', fontWeight: 600,
                    textTransform: 'uppercase', letterSpacing: '0.08em'
                }}>
                    {label}
                </span>
                <span style={{ color: '#6366F1', fontSize: '0.68rem', fontFamily: 'monospace' }}>
                    {index} / {max}
                </span>
            </div>

            <div style={{
                width: '100%', aspectRatio: '1', background: '#000',
                borderRadius: '10px', overflow: 'hidden', position: 'relative',
                border: '1px solid rgba(99,102,241,0.25)',
                boxShadow: '0 0 12px rgba(99,102,241,0.1)',
            }}>
                {loading && (
                    <div style={{
                        position: 'absolute', inset: 0, display: 'flex',
                        alignItems: 'center', justifyContent: 'center', background: '#000'
                    }}>
                        <div style={{
                            width: '24px', height: '24px', borderRadius: '50%',
                            border: '2px solid rgba(99,102,241,0.3)',
                            borderTopColor: '#6366F1',
                            animation: 'mpr-spin 0.8s linear infinite',
                        }} />
                    </div>
                )}
                {imgUrl && (
                    <img src={imgUrl} alt={`${view} ${index}`}
                        style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
                )}
            </div>

            <input
                type="range" min={0} max={max} value={index}
                onChange={e => onChange(Number(e.target.value))}
                style={{ width: '100%', accentColor: '#6366F1', cursor: 'pointer', height: '4px' }}
            />
        </div>
    )
}

export default function MPRViewer({ caseId, layout = 'horizontal' }: MPRViewerProps) {
    const [info, setInfo] = useState<VolumeInfo | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [axialIdx, setAxialIdx] = useState(0)
    const [coronalIdx, setCoronalIdx] = useState(0)
    const [sagittalIdx, setSagittalIdx] = useState(0)

    useEffect(() => {
        getVolumeInfo(caseId)
            .then(data => {
                setInfo(data)
                setAxialIdx(data.yolo.z ?? 0)
                setCoronalIdx(data.yolo.cy ?? 0)
                setSagittalIdx(data.yolo.cx ?? 0)
            })
            .catch(() => setError('MPR viewer unavailable for this case'))
            .finally(() => setLoading(false))
    }, [caseId])

    if (loading) return (
        <div style={{
            padding: '2rem', textAlign: 'center', color: '#94A3B8', fontSize: '0.8rem',
            gridColumn: '1 / -1'
        }}>
            Loading viewer...
        </div>
    )

    if (error || !info) return (
        <div style={{
            padding: '2rem', textAlign: 'center', gridColumn: '1 / -1',
            background: 'rgba(99,102,241,0.05)',
            border: '1px dashed rgba(99,102,241,0.2)', borderRadius: '12px',
        }}>
            <p style={{ color: '#6366F1', fontSize: '0.8rem', fontWeight: 600, margin: '0 0 0.25rem' }}>
                Interactive MPR Viewer
            </p>
            <p style={{ color: '#94A3B8', fontSize: '0.72rem', margin: 0 }}>
                {error || 'No data available'}
            </p>
        </div>
    )

    const [nz, ny, nx] = info.shape

    const panels = (
        <>
            <SlicePanel caseId={caseId} view="axial"
                index={axialIdx} max={nz - 1} label="Axial" onChange={setAxialIdx} />
            <SlicePanel caseId={caseId} view="coronal"
                index={coronalIdx} max={ny - 1} label="Coronal" onChange={setCoronalIdx} />
            <SlicePanel caseId={caseId} view="sagittal"
                index={sagittalIdx} max={nx - 1} label="Sagittal" onChange={setSagittalIdx} />
            <style>{`@keyframes mpr-spin { to { transform: rotate(360deg) } }`}</style>
        </>
    )

    if (layout === 'vertical') {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {panels}
            </div>
        )
    }

    return <>{panels}</>
}