import axios from 'axios';
import { Portfolio, AnalysisResult } from '../types/portfolio';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const portfolioAPI = {
  // Portfolio CRUD
  createPortfolio: (data: Omit<Portfolio, 'id' | 'created_at' | 'updated_at'>) =>
    api.post<Portfolio>('/api/v1/portfolios/', data),
  
  getPortfolios: () =>
    api.get<Portfolio[]>('/api/v1/portfolios/'),
  
  getPortfolio: (id: string) =>
    api.get<Portfolio>(`/api/v1/portfolios/${id}`),
  
  updatePortfolio: (id: string, data: Partial<Portfolio>) =>
    api.put<Portfolio>(`/api/v1/portfolios/${id}`, data),
  
  deletePortfolio: (id: string) =>
    api.delete(`/api/v1/portfolios/${id}`),
  
  // Analysis
  runAnalysis: (data: {
    portfolio_id: string;
    analysis_types: string[];
    start_date?: string;
  }) =>
    api.post<AnalysisResult>('/api/v1/analysis/', data),
  
  getAnalysisHistory: (portfolioId: string) =>
    api.get(`/api/v1/analysis/${portfolioId}/history`),
};

export const authAPI = {
  login: (email: string, password: string) =>
    api.post('/api/v1/auth/login', {
      username: email,
      password: password
    }),
  
  register: (email: string, password: string, full_name: string) =>
    api.post('/api/v1/auth/register', {
      email,
      password,
      full_name
    }),
};

export default api;