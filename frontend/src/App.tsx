import { useState } from 'react'
import PDCalculator from './components/PDCalculator'
import StressDashboard from './components/StressDashboard'
import ModelMetricsView from './components/ModelMetrics'

type Page = 'pd' | 'stress' | 'metrics'

export default function App() {
  const [page, setPage] = useState<Page>('pd')

  return (
    <div className="app">
      <nav className="nav">
        <h1>Credit Risk PD Engine</h1>
        <button className={page === 'pd' ? 'active' : ''} onClick={() => setPage('pd')}>PD Calculator</button>
        <button className={page === 'stress' ? 'active' : ''} onClick={() => setPage('stress')}>Stress Test</button>
        <button className={page === 'metrics' ? 'active' : ''} onClick={() => setPage('metrics')}>Model Metrics</button>
      </nav>
      <main className="main">
        {page === 'pd' && <PDCalculator />}
        {page === 'stress' && <StressDashboard />}
        {page === 'metrics' && <ModelMetricsView />}
      </main>
    </div>
  )
}
