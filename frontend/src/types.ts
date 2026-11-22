export type Sku = {
  sku_id: string;
  total_units: number;
  avg_daily_demand: number;
  unit_price: number;
  zero_sale_rate: number;
  revenue: number;
};

export type Portfolio = {
  total_skus: number;
  total_units: number;
  avg_daily_demand: number;
  total_revenue: number;
  zero_sale_rate: number;
  data_source: string;
  model_available: boolean;
};

export type SkuResponse = {
  portfolio: Portfolio;
  metrics: { wape: number | null; rmsle: number | null };
  metrics_source: string;
  skus: Sku[];
};

export type ForecastPoint = {
  date: string;
  actual?: number | null;
  prediction?: number | null;
  p10?: number | null;
  p90?: number | null;
  ols?: number | null;
  poisson?: number | null;
  zero_sale?: boolean;
};

export type ForecastResponse = {
  sku_id: string;
  horizon: number;
  model: string;
  model_available: boolean;
  points: ForecastPoint[];
};

export type PriceInput = {
  sku_id: string;
  unit_cost: number;
  competitor_index: number;
  min_margin_floor: number;
  max_price_delta_pct: number;
};

export type ProfitPoint = {
  price: number;
  demand: number;
  profit: number;
  feasible: boolean;
};

export type PriceResponse = {
  sku_id: string;
  optimal_price: number;
  projected_demand: number;
  net_profit_uplift: number;
  reference_price: number;
  profit_curve: ProfitPoint[];
};

export type Diagnostics = {
  model_available: boolean;
  feature_importance: { feature: string; gain: number }[];
  benchmarks: {
    model: string;
    objective: string;
    wape: number | null;
    rmsle: number | null;
    status: string;
  }[];
  formula: string;
};
