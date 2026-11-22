import type {
  Diagnostics,
  ForecastPoint,
  ForecastResponse,
  Portfolio,
  PriceInput,
  PriceResponse,
  Sku,
  SkuResponse,
} from "./types";

const API = "http://localhost:8000/api";

const demoSkus: Sku[] = [
  ["85123A", 42.8, 2.4, 14.2, 0.71],
  ["22423", 31.6, 8.7, 42.9, 0.48],
  ["85099B", 24.1, 3.2, 18.6, 0.62],
  ["21212", 19.8, 1.6, 12.4, 0.76],
  ["POST", 16.4, 5.2, 30.1, 0.54],
  ["47566", 13.2, 7.1, 37.2, 0.43],
  ["22457", 10.7, 4.8, 22.5, 0.59],
  ["20725", 8.9, 2.9, 16.8, 0.67],
].map(([sku_id, total_units, avg_daily_demand, unit_price, zero_sale_rate]) => ({
  sku_id: String(sku_id),
  total_units: Number(total_units) * 1000,
  avg_daily_demand: Number(avg_daily_demand),
  unit_price: Number(unit_price),
  zero_sale_rate: Number(zero_sale_rate),
  revenue: Number(total_units) * Number(unit_price) * 1000,
}));

const demoPortfolio: Portfolio = {
  total_skus: 148,
  total_units: 284600,
  avg_daily_demand: 780,
  total_revenue: 1842500,
  zero_sale_rate: 0.57,
  data_source: "offline simulation",
  model_available: false,
};

let online = true;
export const connectionState = () => online;

async function request<T>(path: string, init?: RequestInit): Promise<T | null> {
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      signal: AbortSignal.timeout(3500),
    });
  } catch (error) {
    if (error instanceof TypeError || (error instanceof DOMException && error.name === "TimeoutError")) {
      online = false;
      return null;
    }
    throw error;
  }
  if (!response.ok) throw new Error(`API ${response.status} for ${path}`);
  online = true;
  return (await response.json()) as T;
}

export async function fetchSkus(): Promise<SkuResponse> {
  return (
    (await request<SkuResponse>("/skus")) ?? {
      portfolio: demoPortfolio,
      metrics: { wape: 0.184, rmsle: 0.692 },
      metrics_source: "simulation",
      skus: demoSkus,
    }
  );
}

function makeMockForecast(skuId: string, horizon: number): ForecastResponse {
  const sku = demoSkus.find((item) => item.sku_id === skuId) ?? demoSkus[0];
  const points: ForecastPoint[] = Array.from({ length: 60 + horizon }, (_, index) => {
    const date = new Date();
    date.setDate(date.getDate() - 59 + index);
    const season = Math.sin(index * 0.32) * sku.avg_daily_demand * 0.25;
    const noise = Math.sin(index * 1.73 + sku.unit_price) * sku.avg_daily_demand * 0.24;
    const mean = sku.avg_daily_demand + season + noise;
    if (index < 60) {
      const zero = Math.sin(index * 2.11 + sku.total_units) > 0.68;
      return {
        date: date.toISOString().slice(0, 10),
        actual: zero ? 0 : Math.max(mean, 0),
        zero_sale: zero,
      };
    }
    const forecast = Math.max(sku.avg_daily_demand + season * 0.7, 0);
    return {
      date: date.toISOString().slice(0, 10),
      prediction: forecast,
      p10: Math.max(forecast * 0.64, 0),
      p90: forecast * 1.39,
      ols: Math.max(forecast * (1 + Math.cos(index) * 0.08), 0),
      poisson: Math.max(forecast * (1 + Math.sin(index * 0.7) * 0.06), 0),
    };
  });
  return {
    sku_id: skuId,
    horizon,
    model: "Poisson live-data fallback (demo)",
    model_available: false,
    points,
  };
}

export async function fetchForecast(skuId: string, horizon: number) {
  return (
    (await request<ForecastResponse>(`/forecast/${encodeURIComponent(skuId)}?horizon=${horizon}`)) ??
    makeMockForecast(skuId, horizon)
  );
}

export async function optimizePrice(input: PriceInput): Promise<PriceResponse> {
  const live = await request<PriceResponse>("/optimize-price", {
    method: "POST",
    body: JSON.stringify(input),
  });
  if (live) return live;
  const sku = demoSkus.find((item) => item.sku_id === input.sku_id) ?? demoSkus[0];
  const reference = sku.unit_price * input.competitor_index;
  const low = Math.max(input.unit_cost * (1 + input.min_margin_floor), reference * (1 - input.max_price_delta_pct));
  const high = reference * (1 + input.max_price_delta_pct);
  const elasticity = 1.25;
  const demandAt = (price: number) =>
    Math.min(
      sku.avg_daily_demand * (reference / price) ** elasticity,
      sku.avg_daily_demand * 2,
    );
  const profitCurve = Array.from({ length: 81 }, (_, index) => {
    const price = low + ((high - low) * index) / 80;
    const demand = demandAt(price);
    return { price, demand, profit: (price - input.unit_cost) * demand, feasible: true };
  });
  const best = profitCurve.reduce((left, right) => (right.profit > left.profit ? right : left));
  const baseline = (reference - input.unit_cost) * demandAt(reference);
  return {
    sku_id: input.sku_id,
    optimal_price: best.price,
    projected_demand: best.demand,
    net_profit_uplift: best.profit - baseline,
    reference_price: reference,
    profit_curve: profitCurve,
  };
}

export async function fetchDiagnostics(): Promise<Diagnostics> {
  return (
    (await request<Diagnostics>("/diagnostics")) ?? {
      model_available: false,
      formula: "Q(P) = Q_ref × (P_ref / P)^|ε| ; Π(P) = (P − COGS) × min(Q(P), inventory)",
      feature_importance: [
        ["rolling_7d_mean_qty", 29.4],
        ["lag_1_days", 20.8],
        ["rolling_30d_mean_qty", 16.1],
        ["ewma_7d_qty", 12.7],
        ["day_of_week", 8.6],
        ["relative_price_ratio", 5.8],
        ["days_since_last_sale", 3.7],
        ["month_sin", 2.9],
      ].map(([feature, gain]) => ({ feature: String(feature), gain: Number(gain) })),
      benchmarks: [
        { model: "LightGBM Tweedie", objective: "Tweedie p=1.5", wape: 0.184, rmsle: 0.692, status: "mock metrics" },
        { model: "Poisson GLM", objective: "Poisson deviance", wape: 0.231, rmsle: 0.811, status: "benchmark" },
        { model: "OLS trend", objective: "Squared error", wape: 0.306, rmsle: 0.943, status: "benchmark" },
        { model: "Seasonal naive", objective: "7-day seasonal", wape: 0.267, rmsle: 0.856, status: "benchmark" },
      ],
    }
  );
}

export function exportPricingCsv(skus: Sku[], results: PriceResponse[]) {
  const bySku = new Map(results.map((result) => [result.sku_id, result]));
  const rows = [
    ["sku_id", "current_price", "optimal_price", "projected_daily_demand", "net_profit_uplift"],
    ...skus.map((sku) => {
      const result = bySku.get(sku.sku_id);
      return [
        sku.sku_id,
        sku.unit_price.toFixed(2),
        result?.optimal_price.toFixed(2) ?? "",
        result?.projected_demand.toFixed(2) ?? "",
        result?.net_profit_uplift.toFixed(2) ?? "",
      ];
    }),
  ];
  const blob = new Blob([rows.map((row) => row.join(",")).join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "smart-retail-intelligence-pricing-sheet.csv";
  link.click();
  URL.revokeObjectURL(url);
}
