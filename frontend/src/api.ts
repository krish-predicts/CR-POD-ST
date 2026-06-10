import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const client = axios.create({ baseURL: API_URL, timeout: 120000 })

export interface LoanInput {
  person_age: number
  person_income: number
  person_home_ownership: string
  person_emp_length: number
  loan_intent: string
  loan_grade: string
  loan_amnt: number
  loan_int_rate: number
  loan_percent_income: number
  cb_person_default_on_file: string
  cb_person_cred_hist_length: number
}

export interface Prediction {
  pd: number
  risk_score: number
  risk_band: string
  predicted_class: number
  expected_loss: number
  ifrs9_stage: number
  ecl: number
  model_name: string
}

export interface ModelMetrics {
  model_name: string
  roc_auc: number
  f1: number
  precision: number
  recall: number
  accuracy: number
  ks: number
  gini: number
}

export interface StressResult {
  scenario: string
  avg_pd: number
  total_el: number
  total_ecl: number
  stage_1_count: number
  stage_2_count: number
  stage_3_count: number
  pd_multiplier: number
}

export const api = {
  health: () => client.get('/health'),
  train: () => client.post('/train', {}),
  metrics: () => client.get('/metrics'),
  predict: (loan: LoanInput, model_name = 'xgb') =>
    client.post<{ predictions: Prediction[] }>('/predict', { loan, model_name }),
  stressTest: (sample_size: number, scenarios: string[], model_name = 'xgb') =>
    client.post<{ results: StressResult[]; comparison: unknown[]; loan_count: number }>(
      '/stress_test',
      { sample_size, scenarios, model_name },
    ),
}
