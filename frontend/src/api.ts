import type {
  Diagnostics,
  ForecastResponse,
  IngestionAudit,
  PriceInput,
  PriceResponse,
  ScreenerResponse,
  SkuResponse,
} from "./types";

const API = "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 10000;
const RETRY_DELAYS_MS = [500, 1000, 2000, 4000];

let online = false;
export const connectionState = () => online;

const delay = (milliseconds: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = REQUEST_TIMEOUT_MS,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    online = false;
    const detail = error instanceof Error ? error.message : "Network request failed.";
    throw new Error(`Live API unavailable for ${path}: ${detail}`);
  }

  if (!response.ok) {
    online = response.status < 500;
    let detail = "";
    try {
      const body = await response.json() as { detail?: string | { msg?: string }[] };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail.map((item) => item.msg).filter(Boolean).join("; ");
      }
    } catch {
      detail = "";
    }
    throw new Error(
      detail ? `API ${response.status} for ${path}: ${detail}` : `API ${response.status} for ${path}`,
    );
  }

  online = true;
  return await response.json() as T;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const health = await request<{ status: string }>("/health", undefined, 1000);
    online = health.status === "ok";
    return online;
  } catch {
    return false;
  }
}

export async function connectToLiveApi(): Promise<{
  portfolio: SkuResponse;
  diagnostics: Diagnostics;
}> {
  let lastError: Error = new Error("The live API is not ready.");
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      if (!await checkHealth()) throw new Error("FastAPI health check is not ready.");
      const [portfolio, diagnostics] = await Promise.all([
        fetchSkus(),
        fetchDiagnostics(),
      ]);
      if (
        portfolio.mode !== "live_model" ||
        !portfolio.portfolio.model_available ||
        portfolio.portfolio.total_skus <= 8
      ) {
        throw new Error("The API did not return the trained live portfolio.");
      }
      return { portfolio, diagnostics };
    } catch (error) {
      online = false;
      lastError = error instanceof Error ? error : new Error("Live API connection failed.");
      if (attempt < RETRY_DELAYS_MS.length) await delay(RETRY_DELAYS_MS[attempt]);
    }
  }
  throw lastError;
}

export async function fetchSkus(): Promise<SkuResponse> {
  return request<SkuResponse>("/api/skus");
}

export async function fetchScreener(): Promise<ScreenerResponse> {
  return request<ScreenerResponse>("/api/screener");
}

export async function fetchForecast(skuId: string, horizon: number): Promise<ForecastResponse> {
  return request<ForecastResponse>(
    `/api/forecast/${encodeURIComponent(skuId)}?horizon=${horizon}`,
  );
}

export async function optimizePrice(input: PriceInput): Promise<PriceResponse> {
  const payload: PriceInput = {
    sku_id: input.sku_id,
    unit_cost: Number(input.unit_cost.toFixed(2)),
    competitor_index: Number(input.competitor_index.toFixed(4)),
    min_margin_floor: Number(input.min_margin_floor.toFixed(4)),
    max_price_delta_pct: Number(input.max_price_delta_pct.toFixed(4)),
    inventory: Number(input.inventory.toFixed(2)),
    elasticity: Number(input.elasticity.toFixed(4)),
    strict_bounds: input.strict_bounds ?? false,
  };
  return request<PriceResponse>("/api/optimize-price", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchDiagnostics(): Promise<Diagnostics> {
  return request<Diagnostics>("/api/diagnostics");
}

export async function uploadCsv(file: File): Promise<IngestionAudit> {
  const query = new URLSearchParams({ filename: file.name });
  return request<IngestionAudit>(`/api/ingest?${query}`, {
    method: "POST",
    headers: { "Content-Type": "text/csv" },
    body: file,
  });
}

export function exportPricingCsv(skus: SkuResponse["skus"], results: PriceResponse[]) {
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
