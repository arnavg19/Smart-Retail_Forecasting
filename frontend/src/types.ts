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
  mode: string;
};

export type SkuResponse = {
  portfolio: Portfolio;
  metrics: {
    wape: number | null;
    rmsle: number | null;
    baseline_wape: number | null;
    baseline_rmsle: number | null;
  };
  metrics_source: string;
  skus: Sku[];
  mode: string;
};

export type ScreenerResponse = {
  total_skus: number;
  limit: number;
  skus: Sku[];
};

export type ForecastPoint = {
  date: string;
  actual?: number | null;
  prediction?: number | null;
  p10?: number | null;
  p90?: number | null;
  baseline?: number | null;
  zero_sale?: boolean;
};

export type ForecastResponse = {
  sku_id: string;
  horizon: number;
  model: string;
  model_available: boolean;
  points: ForecastPoint[];
  mode: string;
};

export type PriceInput = {
  sku_id: string;
  unit_cost: number;
  competitor_index: number;
  min_margin_floor: number;
  max_price_delta_pct: number;
  inventory: number;
  elasticity: number;
  strict_bounds?: boolean;
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
  inventory_cap: number;
  unit_cost: number;
  min_margin_floor: number;
  historical_max_price: number;
  max_price_delta_pct: number;
  elasticity: number;
  elasticity_source: string;
  mode: string;
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
  mode: string;
  tweedie_variance_power: number | null;
};

export type IngestionAudit = {
  input_rows: number;
  valid_transaction_rows: number;
  dropped_invalid_rows: number;
  active_skus: number;
  calendar_periods: number;
  grid_rows: number;
  zero_sale_rows: number;
  zero_sale_grid_expansion_rows: number;
  preview: {
    item_id: string;
    invoice_date: string;
    quantity: number;
    unit_price: number;
    gross_revenue: number;
  }[];
  mode: string;
};
