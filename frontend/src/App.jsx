import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import './App.css'

const API = import.meta.env.VITE_API_URL

export default function App() {
  const [uploadStatus, setUploadStatus] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [documents, setDocuments] = useState([])
  const [deleting, setDeleting] = useState(null)
  const [question, setQuestion] = useState('')
  const [querying, setQuerying] = useState(false)
  const [runEval, setRunEval] = useState(false)
  const [result, setResult] = useState(null)
  const [queryError, setQueryError] = useState(null)
  const [history, setHistory] = useState([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const fileRef = useRef()

  function fetchDocuments() {
    fetch(`${API}/api/documents`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => setDocuments(Array.isArray(data) ? data : []))
      .catch(() => {})
  }

  async function handleDelete(docName) {
    setDeleting(docName)
    try {
      await fetch(`${API}/api/documents/${encodeURIComponent(docName)}`, { method: 'DELETE' })
      fetchDocuments()
    } catch (_) {}
    finally { setDeleting(null) }
  }

  useEffect(() => {
    fetchDocuments()
    fetch(`${API}/api/history`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => setHistory(Array.isArray(data) ? data : []))
      .catch(() => {})
  }, [result])

  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    setUploading(true)
    setUploadStatus(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch(`${API}/api/documents`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail?.error || 'Upload failed')
      setUploadStatus({ ok: true, chunk_count: data.chunk_count, pages_skipped: data.pages_skipped })
      fetchDocuments()
    } catch (err) {
      setUploadStatus({ ok: false, error: err.message })
    } finally {
      setUploading(false)
      fileRef.current.value = ''
    }
  }

  async function handleQuery(e) {
    e.preventDefault()
    if (!question.trim()) return
    setQuerying(true)
    setResult(null)
    setQueryError(null)
    try {
      const res = await fetch(`${API}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, run_evaluation: runEval }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail?.error || 'Query failed')
      setResult(data)
    } catch (err) {
      setQueryError(err.message)
    } finally {
      setQuerying(false)
    }
  }

  function fmt(val) {
    return val != null ? `${(val * 100).toFixed(0)}%` : '—'
  }

  function preprocessMath(text) {
    if (!text) return text
    return text
      .replace(/\\\[/g, '\n$$\n')
      .replace(/\\\]/g, '\n$$\n')
      .replace(/\\\(/g, '$')
      .replace(/\\\)/g, '$')
  }

  return (
    <div className="app">
      <header>
        <h1>RAG Pipeline</h1>
        <p className="subtitle">Self-RAG with Ragas evaluation</p>
      </header>

      {/* Upload + Documents side by side */}
      <div className="top-row">
        <section className="panel upload-panel">
          <h2>Upload Document</h2>
          <div className="upload-area">
            <input ref={fileRef} type="file" accept=".pdf" onChange={handleUpload} disabled={uploading} />
            {uploading && <p className="status loading">Parsing and embedding document...</p>}
            {uploadStatus?.ok && (
              <p className="status success">
                Uploaded — {uploadStatus.chunk_count} chunks stored
                {uploadStatus.pages_skipped > 0 && `, ${uploadStatus.pages_skipped} pages skipped`}
              </p>
            )}
            {uploadStatus?.ok === false && <p className="status error">{uploadStatus.error}</p>}
          </div>
        </section>

        <section className="panel docs-panel">
          <h2>Indexed Documents</h2>
          {documents.length === 0 ? (
            <p className="empty-state">No documents indexed yet.</p>
          ) : (
            <ul className="doc-list">
              {documents.map((d, i) => (
                <li key={i} className="doc-row">
                  <span className="doc-name">{d.document}</span>
                  <span className="doc-row-right">
                    <span className="doc-chunks">{d.chunk_count} chunks</span>
                    <button
                      className="doc-delete"
                      onClick={() => handleDelete(d.document)}
                      disabled={deleting === d.document}
                      title="Delete document"
                    >
                      {deleting === d.document ? '…' : '✕'}
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      {/* Query panel */}
      <section className="panel">
        <h2>Ask a Question</h2>
        <form onSubmit={handleQuery} className="query-form">
          <input
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="e.g. What is EIC's revenue in 2025?"
            disabled={querying}
          />
          <button type="submit" disabled={querying || !question.trim()}>
            {querying ? (runEval ? 'Evaluating...' : 'Thinking...') : 'Ask'}
          </button>
        </form>
        <label className="eval-toggle">
          <input type="checkbox" checked={runEval} onChange={e => setRunEval(e.target.checked)} />
          Run Ragas evaluation <span className="eval-note">(adds ~20s)</span>
        </label>

        {queryError && <p className="status error">{queryError}</p>}

        {result && (
          <div className="result">
            {result.low_confidence && (
              <p className="low-confidence">Low confidence — retrieved context may be insufficient</p>
            )}
            <div className="answer">
              <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{preprocessMath(result.answer)}</ReactMarkdown>
            </div>

            {result.sources?.length > 0 && (
              <div className="sources">
                <h4>Sources</h4>
                {result.sources.map((s, i) => (
                  <span key={i} className="source-tag">
                    {s.document} · p.{s.page} [{s.type}]
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      {/* Metrics panel */}
      <section className="panel">
        <h2>Ragas Metrics</h2>
        <div className="metrics-grid">
          <div className={`metric ${!result ? 'empty' : ''}`}>
            <span className="metric-label">Faithfulness</span>
            <span className="metric-value">{result ? fmt(result.faithfulness) : '—'}</span>
          </div>
          <div className={`metric ${!result ? 'empty' : ''}`}>
            <span className="metric-label">Answer Relevancy</span>
            <span className="metric-value">{result ? fmt(result.answer_relevancy) : '—'}</span>
          </div>
          <div className={`metric ${!result ? 'empty' : ''}`}>
            <span className="metric-label">Context Recall</span>
            <span className="metric-value">{result ? fmt(result.context_recall) : '—'}</span>
          </div>
          <div className={`metric ${!result ? 'empty' : ''}`}>
            <span className="metric-label">Latency</span>
            <span className="metric-value">{result ? `${result.latency_ms}ms` : '—'}</span>
          </div>
        </div>
      </section>

      {/* History panel */}
      <section className="panel">
        <h2 className="collapsible" onClick={() => setHistoryOpen(o => !o)}>
          Query History {historyOpen ? '▲' : '▼'}
        </h2>
        {historyOpen && (
          <div className="history">
            {history.length === 0 && <p className="empty-state">No queries yet.</p>}
            {history.map((h, i) => (
              <div key={i} className="history-row">
                <div className="history-query">{h.query}</div>
                <div className="history-meta">
                  <span className={h.low_confidence ? 'badge low' : 'badge ok'}>
                    {h.low_confidence ? 'low confidence' : 'ok'}
                  </span>
                  <span>F: {fmt(h.faithfulness)}</span>
                  <span>AR: {fmt(h.answer_relevancy)}</span>
                  <span>CR: {fmt(h.context_recall)}</span>
                  <span>{h.latency_ms}ms</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
