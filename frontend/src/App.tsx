import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent, ReactNode } from "react";
import {
  Activity,
  ArrowDownUp,
  ArrowUpRight,
  BarChart3,
  Bell,
  CalendarDays,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Download,
  Layers,
  LayoutDashboard,
  ListFilter,
  LoaderCircle,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Moon,
  Sun,
  Tag,
  Waves,
  Zap,
} from "lucide-react";
import {
  checkHealth,
  connectionState,
  connectToLiveApi,
  exportPricingCsv,
  fetchDiagnostics,
  fetchForecast,
  fetchScreener,
  optimizePrice,
} from "./api";
import type {
  Diagnostics,
  ForecastPoint,
  ForecastResponse,
  PriceInput,
  PriceResponse,
  ScreenerResponse,
  Sku,
  SkuResponse,
} from "./types";

type Workspace = "overview" | "pricing" | "explainability" | "screener";

const views: { id: Workspace; number: string; label: string; icon: typeof Activity }[] = [
  { id: "overview", number: "01", label: "Forecast Telemetry", icon: Activity },
  { id: "pricing", number: "02", label: "Price Optimizer", icon: SlidersHorizontal },
  { id: "explainability", number: "03", label: "Model Attribution", icon: BarChart3 },
  { id: "screener", number: "04", label: "Portfolio Screener", icon: ListFilter },
];

const initialPricing: Omit<PriceInput, "sku_id"> = {
  unit_cost: 9.4,
  competitor_index: 1,
  min_margin_floor: 0.18,
  max_price_delta_pct: 0.25,
  inventory: 1000,
  elasticity: 1.5,
};

const EMPTY_FORECAST_POINTS: ForecastPoint[] = [];
const EMPTY_PROFIT_CURVE: PriceResponse["profit_curve"] = [];

const money = (value: number, decimals = 2) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);

const number = (value: number, decimals = 1) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: decimals }).format(value);

function useDebouncedValue<T>(value: T, delayMs = 150): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

const CALCULATION_FEEDBACK_MS = 600;

async function waitForCalculationFeedback(startedAt: number): Promise<void> {
  const remaining = CALCULATION_FEEDBACK_MS - (performance.now() - startedAt);
  if (remaining > 0) {
    await new Promise<void>((resolve) => window.setTimeout(resolve, remaining));
  }
}

function CalculatingStatus() {
  return (
    <span className="calculating-status" role="status" aria-live="polite">
      <LoaderCircle className="spin" size={13} />
      Calculating...
    </span>
  );
}

function Metric({
  label,
  value,
  change,
  tone = "neutral",
  icon,
}: {
  label: string;
  value: string;
  change?: string;
  tone?: "positive" | "cyan" | "neutral";
  icon?: ReactNode;
}) {
  return (
    <div className="metric-cell">
      {icon && <span className="metric-icon">{icon}</span>}
      <div className="metric-body">
        <div className="metric-label">{label}</div>
        <div className={`metric-value ${tone}`}>{value}</div>
        {change && <div className="metric-change">{change}</div>}
      </div>
    </div>
  );
}

function PanelTitle({
  eyebrow,
  title,
  suffix,
  action,
}: {
  eyebrow: string;
  title: string;
  suffix?: string;
  action?: ReactNode;
}) {
  return (
    <div className="panel-title-row">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <div className="panel-heading">{title}{suffix && <span className="title-suffix">{suffix}</span>}</div>
      </div>
      {action}
    </div>
  );
}

function ForecastChart({
  response,
  calculating,
}: {
  response: ForecastResponse | null;
  calculating: boolean;
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const points = response?.points ?? EMPTY_FORECAST_POINTS;
  const historical = useMemo(
    () => points.filter((point) => point.actual != null),
    [points],
  );
  const future = useMemo(
    () => points.filter((point) => point.prediction != null),
    [points],
  );
  const values = useMemo(
    () => points.flatMap((point) => [
      point.actual ?? point.prediction ?? 0,
      point.p10 ?? point.actual ?? 0,
      point.p90 ?? point.actual ?? 0,
    ]),
    [points],
  );
  const maxY = useMemo(() => Math.max(...values, 1) * 1.16, [values]);
  const yTicks = useMemo(
    () => Array.from({ length: 5 }, (_, index) => (maxY * index) / 4),
    [maxY],
  );
  const minY = 0;
  const width = 1000;
  const height = 400;
  const pad = { left: 54, right: 20, top: 20, bottom: 34 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const x = useCallback(
    (index: number) => pad.left + (index / Math.max(points.length - 1, 1)) * plotW,
    [plotW, points.length],
  );
  const y = useCallback(
    (value: number) => pad.top + (1 - (value - minY) / (maxY - minY || 1)) * plotH,
    [maxY, minY, plotH],
  );
  const linePath = (series: Array<number | null | undefined>, offset = 0) =>
    series.reduce((path, value, index) => {
      if (value === null || value === undefined) return path;
      return `${path}${path ? "L" : "M"}${x(index + offset).toFixed(1)},${y(value).toFixed(1)} `;
    }, "");
  const futureOffset = historical.length;
  const paths = useMemo(() => {
    const forecastPath = linePath(future.map((point) => point.prediction), futureOffset);
    const highPath = linePath(future.map((point) => point.p90), futureOffset);
    const bandPath = highPath
      ? `${highPath} ${future.map((point, index) => `L${x(futureOffset + future.length - index - 1).toFixed(1)},${y(point.p10 ?? 0).toFixed(1)}`).join(" ")} Z`
      : "";
    return {
      forecastPath,
      bandPath,
      historicalPath: linePath(historical.map((point) => point.actual)),
    };
  }, [future, futureOffset, historical, x, y]);
  const active = useMemo(
    () => hoverIndex === null ? null : points[hoverIndex],
    [hoverIndex, points],
  );
  const onMove = useCallback((event: MouseEvent<SVGSVGElement>) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box) return;
    const relative = ((event.clientX - box.left) / box.width) * width;
    const fraction = Math.max(0, Math.min(1, (relative - pad.left) / plotW));
    setHoverIndex(Math.round(fraction * (points.length - 1)));
  }, [pad.left, plotW, points.length]);
  if (!response) {
    return (
      <div className="chart-empty">
        {calculating ? <CalculatingStatus /> : <span>Forecast unavailable</span>}
      </div>
    );
  }
  return (
    <div className="chart-wrap">
      <svg
        ref={svgRef}
        className="forecast-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Historical demand and future SKU demand forecast"
        onMouseMove={onMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <defs>
          <linearGradient id="confidence-band" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#A78BFA" stopOpacity=".16" />
            <stop offset="100%" stopColor="#A78BFA" stopOpacity=".015" />
          </linearGradient>
        </defs>
        {yTicks.map((tick, index) => (
          <g key={index}>
            <line x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
            <text x={pad.left - 10} y={y(tick) + 4} textAnchor="end" className="axis-label">{number(tick, 0)}</text>
          </g>
        ))}
        <line x1={x(futureOffset)} x2={x(futureOffset)} y1={pad.top} y2={height - pad.bottom} className="forecast-divider" />
        {paths.bandPath && <path d={paths.bandPath} fill="url(#confidence-band)" />}
        <path d={paths.historicalPath} className="line-history" />
        <path d={paths.forecastPath} className="line-forecast" />
        {historical.map((point, index) =>
          point.zero_sale ? (
            <circle key={`zero-${index}`} cx={x(index)} cy={y(0)} r="2.8" className="zero-marker" />
          ) : null,
        )}
        {hoverIndex !== null && active && (
          <g>
            <line x1={x(hoverIndex)} x2={x(hoverIndex)} y1={pad.top} y2={height - pad.bottom} className="crosshair" />
            <circle
              cx={x(hoverIndex)}
              cy={y(active.actual ?? active.prediction ?? 0)}
              r="4.3"
              className={active.actual !== undefined ? "cross-dot-actual" : "cross-dot-forecast"}
            />
          </g>
        )}
        <text x={pad.left} y={height - 8} className="axis-label">{historical[0]?.date}</text>
        <text x={x(futureOffset)} y={height - 8} className="axis-label">FORECAST →</text>
        <text x={width - pad.right} y={height - 8} textAnchor="end" className="axis-label">{points[points.length - 1]?.date}</text>
      </svg>
      {active && hoverIndex !== null && (
        <div className="chart-tooltip" style={{ left: `${(x(hoverIndex) / width) * 100}%` }}>
          <div className="tooltip-date">{active.date}</div>
          {active.actual != null ? (
            <div className="tooltip-value">Observed <b>{number(active.actual, 2)}</b> units{active.zero_sale && <span className="crimson"> · ZERO</span>}</div>
          ) : (
            <>
              <div className="tooltip-value">Forecast <b>{number(active.prediction ?? 0, 2)}</b></div>
              <div className="tooltip-muted">P10–P90 {number(active.p10 ?? 0)} — {number(active.p90 ?? 0)}</div>
            </>
          )}
        </div>
      )}
      <div className="chart-legend">
        <span><i className="legend-key key-history" />ACTUAL</span>
        <span><i className="legend-key key-forecast" />TWEEDIE FORECAST</span>
        <span><i className="legend-key key-band" />P10—P90</span>
        <span><i className="legend-key key-zero" />ZERO-SALE DAY</span>
      </div>
    </div>
  );
}

function PortfolioMix({ skus }: { skus: Sku[] }) {
  const total = skus.reduce((sum, sku) => sum + sku.total_units, 0) || 1;
  const high = Math.max(Math.ceil(skus.length * 0.1), 1);
  const core = Math.max(Math.ceil(skus.length * 0.5), high);
  const tiers = [
    { name: "High velocity", color: "#c4b5fd", items: skus.slice(0, high) },
    { name: "Core SKUs", color: "#8b5cf6", items: skus.slice(high, core) },
    { name: "Long tail", color: "#60a5fa", items: skus.slice(core) },
  ].map((tier) => ({ ...tier, share: tier.items.reduce((sum, sku) => sum + sku.total_units, 0) / total }));
  const circumference = 2 * Math.PI * 52;
  let offset = 0;
  return (
    <section className="panel mini-panel">
      <div className="mini-title">Portfolio mix</div>
      <div className="mix-body">
        <svg className="donut" viewBox="0 0 140 140" role="img" aria-label="Share of units sold by SKU tier">
          <circle className="donut-track" cx="70" cy="70" r="52" />
          {tiers.map((tier) => {
            const length = tier.share * circumference;
            const segment = (
              <circle key={tier.name} className="donut-seg" cx="70" cy="70" r="52" stroke={tier.color} strokeDasharray={`${length} ${circumference - length}`} strokeDashoffset={-offset} transform="rotate(-90 70 70)" />
            );
            offset += length;
            return segment;
          })}
          <text className="donut-num" x="70" y="72">{skus.length}</text>
          <text className="donut-cap" x="70" y="88">SKUs</text>
        </svg>
        <div className="mix-legend">
          {tiers.map((tier) => <div key={tier.name}><i style={{ background: tier.color }} />{tier.name}<b>{(tier.share * 100).toFixed(0)}%</b></div>)}
        </div>
      </div>
      <div className="mix-note">Share of units sold by the top 10% of SKUs, the next 40%, and the remaining 50%.</div>
    </section>
  );
}

function ModelPerformance({
  metrics,
  modelAvailable,
  metricsSource,
}: {
  metrics: SkuResponse["metrics"];
  modelAvailable: boolean;
  metricsSource: string;
}) {
  return (
    <section className="panel mini-panel">
      <div className="mini-title">Model performance</div>
      <div className="champion-model">
        <b>{modelAvailable ? "LightGBM Tweedie · p=1.5" : "Trained model unavailable"}</b>
        <span>CHAMPION · PERSISTED VALIDATION</span>
      </div>
      <div className="champion-metrics">
        <div><span>WAPE</span><b>{metrics.wape === null ? "—" : `${(metrics.wape * 100).toFixed(1)}%`}</b></div>
        <div><span>RMSLE</span><b>{metrics.rmsle === null ? "—" : metrics.rmsle.toFixed(3)}</b></div>
      </div>
      <div className="champion-notes">
        <p>LightGBM Tweedie uses variance power <b>p = 1.5</b>.</p>
        <p><b>Var(Y) = φ × μ<sup>p</sup></b>; p = 1.5 handles zero-inflated, positive-skew demand.</p>
        <p>Boosted trees model non-linear feature interactions; see <b>Model Attribution</b> for all four benchmarks.</p>
      </div>
      <div className="champion-source">{metricsSource}</div>
    </section>
  );
}

const MemoForecastChart = memo(ForecastChart);
const MemoPortfolioMix = memo(PortfolioMix);
const MemoModelPerformance = memo(ModelPerformance);

function Overview({
  portfolio,
  metrics,
  metricsSource,
  sku,
  horizon,
  setHorizon,
  forecast,
  forecastLoading,
  onSelectSku,
  skus,
}: {
  portfolio: SkuResponse["portfolio"];
  metrics: SkuResponse["metrics"];
  metricsSource: string;
  sku: Sku | undefined;
  horizon: number;
  setHorizon: (value: number) => void;
  forecast: ForecastResponse | null;
  forecastLoading: boolean;
  onSelectSku: (id: string) => void;
  skus: Sku[];
}) {
  const [search, setSearch] = useState("");
  const [watchlistOpen, setWatchlistOpen] = useState(true);
  const [watchlistPage, setWatchlistPage] = useState(0);
  const debouncedSearch = useDebouncedValue(search);
  const filtered = useMemo(
    () => skus.filter((item) => item.sku_id.toLowerCase().includes(debouncedSearch.trim().toLowerCase())),
    [skus, debouncedSearch],
  );
  const watchlistPageCount = Math.max(1, Math.ceil(filtered.length / 50));
  const watchlistRows = useMemo(
    () => filtered.slice(watchlistPage * 50, (watchlistPage + 1) * 50),
    [filtered, watchlistPage],
  );
  useEffect(() => setWatchlistPage(0), [debouncedSearch]);
  const modelBadge = portfolio.model_available && portfolio.mode === "live_model"
    ? "Portfolio live"
    : "Baseline live";
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">Demand intelligence</div><h1>Forecast <em>Telemetry</em></h1><p className="hero-sub">Demand forecasting, zero-inflation analysis, and portfolio intelligence.</p></div>
        <div className="heading-meta"><span className="live-tag"><i />{modelBadge}</span></div>
      </div>
      <section className="metric-strip">
        <Metric icon={<Layers size={22} />} label="Portfolio SKUs" value={number(portfolio.total_skus, 0)} change="Active universe" />
        <Metric icon={<Tag size={22} />} label="Demand / Day" value={number(portfolio.avg_daily_demand, 0)} tone="cyan" change="Units across all SKUs" />
        <Metric icon={<Activity size={22} />} label={portfolio.model_available ? "WAPE · Validation" : "WAPE · Baseline"} value={metrics.wape === null ? "—" : `${(metrics.wape * 100).toFixed(1)}%`} change={`${metrics.rmsle === null ? "RMSLE —" : `RMSLE ${metrics.rmsle.toFixed(3)}`} · ${metricsSource}`} tone={metrics.wape === null ? "neutral" : "positive"} />
        <Metric icon={<Waves size={22} />} label="Zero-Sale Frequency" value={`${number(portfolio.zero_sale_rate * 100, 1)}%`} change="SKU-days with no sales" />
      </section>
      <div className={`workspace-grid ${watchlistOpen ? "" : "watchlist-collapsed"}`}>
        <aside className="panel watchlist-panel">
          <div className="watchlist-head">
            <div><div className="eyebrow">UNIVERSE</div><div className="panel-heading">SKU watchlist <span className="count-chip">{skus.length}</span></div></div>
            <button className="icon-button" aria-label={watchlistOpen ? "Collapse watchlist" : "Expand watchlist"} onClick={() => setWatchlistOpen(!watchlistOpen)}>
              {watchlistOpen ? <PanelLeftClose size={15} /> : <PanelLeftOpen size={15} />}
            </button>
          </div>
          {watchlistOpen && (
            <>
              <label className="search-field"><Search size={14} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search SKU / product" /><span>/</span></label>
              <div className="watchlist-columns"><span>INSTRUMENT</span><span>DEMAND</span></div>
              <div className="watchlist-list">
                {watchlistRows.map((item, index) => (
                  <button key={item.sku_id} className={`watch-item ${sku?.sku_id === item.sku_id ? "selected" : ""}`} onClick={() => onSelectSku(item.sku_id)}>
                    <span className="watch-symbol"><b>{item.sku_id}</b><small>{index < 3 ? "High velocity" : "Core SKU"}</small></span>
                    <span className="watch-demand"><b>{number(item.avg_daily_demand, 1)}</b><small>{(item.zero_sale_rate * 100).toFixed(0)}% zeros</small></span>
                  </button>
                ))}
                {filtered.length === 0 && <div className="empty-note">No matching instruments.</div>}
              </div>
              <div className="watchlist-footer">
                <span><i className="status-dot" />{filtered.length} INSTRUMENTS</span>
                <span className="list-pager">
                  <button aria-label="Previous SKU page" disabled={watchlistPage === 0} onClick={() => setWatchlistPage((page) => page - 1)}>‹</button>
                  {watchlistPage + 1}/{watchlistPageCount}
                  <button aria-label="Next SKU page" disabled={watchlistPage + 1 >= watchlistPageCount} onClick={() => setWatchlistPage((page) => page + 1)}>›</button>
                </span>
              </div>
            </>
          )}
        </aside>
        <section className="panel forecast-panel">
          <PanelTitle
            eyebrow="DEMAND SIGNAL // ZERO-INFLATED DAILY SERIES"
            title={sku?.sku_id ?? "Loading instrument"}
            suffix={sku ? `· ${money(sku.unit_price)} REF` : undefined}
            action={<button className="icon-button" aria-label="Chart help"><CircleHelp size={15} /></button>}
          />
          <div className="chart-toolbar">
            <div className="instrument-detail"><span className="detail-label">SKU SELECTED</span><span>{sku?.sku_id}</span><span className="detail-separator">/</span><span>60D HISTORY</span><ChevronDown size={13} /></div>
            <div className="toolbar-right">
              <div className="segmented">
                {[7, 14, 30].map((days) => <button key={days} className={horizon === days ? "active" : ""} onClick={() => setHorizon(days)}>{days}D</button>)}
              </div>
            </div>
          </div>
          <div className="chart-meta-row">
            <span className="chart-model"><span className="status-dot cyan-dot" />{forecast?.model ?? "Connecting to forecast service"}</span>
            <span>ZERO DAYS: <b className="crimson">{number((sku?.zero_sale_rate ?? 0) * 60, 0)} / 60</b></span>
            <span>FORECAST: <b className="cyan">{horizon} DAY HORIZON</b></span>
          </div>
          <MemoForecastChart response={forecast} calculating={forecastLoading} />
          <div className="forecast-foot">
            <span><span className="foot-key">MODEL</span> {forecast?.model_available ? "LIGHTGBM TWEEDIE · p=1.5" : "BASELINE ONLY · NO MODEL"}</span>
            <span><span className="foot-key">SOURCE</span> {portfolio.data_source}</span>
            <span>ZERO SALES ARE OBSERVED SIGNALS, NOT MISSING DATA</span>
          </div>
        </section>
        <aside className="side-stack">
          <MemoPortfolioMix skus={skus} />
          <MemoModelPerformance metrics={metrics} modelAvailable={portfolio.model_available} metricsSource={metricsSource} />
        </aside>
      </div>
    </>
  );
}

const MemoOverview = memo(Overview);

function Parameter({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="parameter">
      <div className="parameter-head"><label>{label}</label><span className="mono">{display}</span></div>
      <div className="range-input-row">
        <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
        <input className="number-input" type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      </div>
    </div>
  );
}

function ProfitChart({
  result,
  params,
  busy,
  error,
}: {
  result: PriceResponse | null;
  params: Omit<PriceInput, "sku_id">;
  busy: boolean;
  error: string;
}) {
  const values = result?.profit_curve ?? EMPTY_PROFIT_CURVE;
  const maxProfit = useMemo(
    () => Math.max(...values.map((point) => point.profit), 1),
    [values],
  );
  const demandMax = useMemo(
    () => Math.max(...values.map((point) => point.demand), 1),
    [values],
  );
  if (!result || values.length === 0) {
    return (
      <div className="chart-empty">
        {busy ? <CalculatingStatus /> : error ? <span>{error}</span> : <span>Awaiting constrained optimum</span>}
      </div>
    );
  }
  const width = 920;
  const height = 300;
  const pad = { left: 54, right: 26, top: 20, bottom: 35 };
  const xMin = result.reference_price * (1 - params.max_price_delta_pct);
  const xMax = result.reference_price * (1 + params.max_price_delta_pct);
  const lowerFeasible = Math.max(
    result.reference_price * (1 - params.max_price_delta_pct),
    params.unit_cost * (1 + params.min_margin_floor),
  );
  const x = (price: number) => pad.left + ((price - xMin) / (xMax - xMin || 1)) * (width - pad.left - pad.right);
  const y = (profit: number) => pad.top + (1 - profit / (maxProfit * 1.15)) * (height - pad.top - pad.bottom);
  const profitPath = values.map((point, index) => `${index ? "L" : "M"}${x(point.price).toFixed(1)},${y(point.profit).toFixed(1)}`).join(" ");
  const demandY = (demand: number) => pad.top + (1 - demand / (demandMax * 1.18)) * (height - pad.top - pad.bottom);
  const demandPath = values.map((point, index) => `${index ? "L" : "M"}${x(point.price).toFixed(1)},${demandY(point.demand).toFixed(1)}`).join(" ");
  const optimumX = x(result.optimal_price);
  const infeasibleWidth = Math.max(0, x(lowerFeasible) - pad.left);
  return (
    <svg className="profit-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Interactive constrained profit and demand curve">
      <defs><pattern id="hatch" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse"><line x1="0" y1="0" x2="0" y2="7" stroke="#F43F5E" strokeOpacity=".23" strokeWidth="2" /></pattern></defs>
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
        const yy = pad.top + fraction * (height - pad.top - pad.bottom);
        return <g key={fraction}><line x1={pad.left} x2={width - pad.right} y1={yy} y2={yy} className="chart-grid" /><text x={pad.left - 9} y={yy + 4} textAnchor="end" className="axis-label">{number(maxProfit * (1 - fraction), 0)}</text></g>;
      })}
      <rect x={pad.left} y={pad.top} width={infeasibleWidth} height={height - pad.top - pad.bottom} fill="url(#hatch)" />
      <path d={demandPath} className="line-demand" />
      <path d={profitPath} className="line-profit" />
      <line x1={optimumX} x2={optimumX} y1={pad.top} y2={height - pad.bottom} className="optimal-crosshair" />
      <circle cx={optimumX} cy={y(Math.max(...values.map((point) => point.profit)))} r="5" className="optimal-dot" />
      <text x={optimumX + 8} y={pad.top + 13} className="optimal-label">P* {money(result.optimal_price)}</text>
      <text x={xMin} y={height - 9} className="axis-label">{money(xMin, 0)}</text>
      <text x={width - pad.right} y={height - 9} textAnchor="end" className="axis-label">{money(xMax, 0)}</text>
      <text x={pad.left + 5} y={pad.top + 14} className="infeasible-label">INFEASIBLE / MARGIN FLOOR</text>
      <text x={width - pad.right} y={pad.top + 14} textAnchor="end" className="axis-label">DEMAND INDEX ↗</text>
    </svg>
  );
}

const MemoProfitChart = memo(ProfitChart);

function PricingView({
  sku,
  skus,
  onSelectSku,
  result,
  setResult,
}: {
  sku: Sku | undefined;
  skus: Sku[];
  onSelectSku: (id: string) => void;
  result: PriceResponse | null;
  setResult: (value: PriceResponse | null) => void;
}) {
  const [params, setParams] = useState(initialPricing);
  const [paramsSku, setParamsSku] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [skuPickerOpen, setSkuPickerOpen] = useState(false);
  const [skuSearch, setSkuSearch] = useState("");
  const [skuOptionLimit, setSkuOptionLimit] = useState(50);
  const debouncedSkuSearch = useDebouncedValue(skuSearch);
  const filteredSkuOptions = useMemo(
    () => skus.filter((item) =>
      item.sku_id.toLowerCase().includes(debouncedSkuSearch.trim().toLowerCase()),
    ),
    [skus, debouncedSkuSearch],
  );
  const requestSequence = useRef(0);
  useEffect(() => setSkuOptionLimit(50), [debouncedSkuSearch]);
  useEffect(() => {
    if (!sku) {
      setBusy(false);
      setResult(null);
      return;
    }
    requestSequence.current += 1;
    setBusy(true);
    setResult(null);
    setError("");
    setParams({
      ...initialPricing,
      unit_cost: Number(Math.max(sku.unit_price * 0.58, 0.1).toFixed(2)),
      inventory: Math.max(Math.ceil(sku.avg_daily_demand * 30), 1),
      elasticity: 1.5,
    });
    setParamsSku(sku.sku_id);
  }, [sku?.sku_id, sku?.unit_price]);
  const run = useCallback(async () => {
    if (!sku) return;
    if (paramsSku !== sku.sku_id) return;
    const sequence = ++requestSequence.current;
    const startedAt = performance.now();
    setBusy(true);
    setError("");
    setResult(null);
    let nextResult: PriceResponse | null = null;
    let nextError = "";
    try {
      nextResult = await optimizePrice({
        sku_id: sku.sku_id,
        unit_cost: Number(params.unit_cost.toFixed(2)),
        competitor_index: Number(params.competitor_index.toFixed(4)),
        min_margin_floor: Number(params.min_margin_floor.toFixed(4)),
        max_price_delta_pct: Number(params.max_price_delta_pct.toFixed(4)),
        inventory: Number(params.inventory.toFixed(2)),
        elasticity: Number(params.elasticity.toFixed(4)),
        strict_bounds: false,
      });
    } catch (reason) {
      nextError = reason instanceof Error
        ? reason.message.includes("No feasible price")
          ? "Policy bounds adjusted: no feasible price in this range."
          : "The optimizer could not complete this request. Adjust the controls and try again."
        : "The optimizer could not complete this request.";
    }
    await waitForCalculationFeedback(startedAt);
    if (sequence !== requestSequence.current) return;
    if (nextResult) setResult(nextResult);
    setError(nextError);
    setBusy(false);
  }, [params, paramsSku, setResult, sku]);
  useEffect(() => {
    const timer = window.setTimeout(() => void run(), 150);
    return () => window.clearTimeout(timer);
  }, [run]);
  const update = useCallback((key: keyof typeof params, value: number) => {
    requestSequence.current += 1;
    setBusy(true);
    setResult(null);
    setError("");
    setParams((current) => ({ ...current, [key]: value }));
  }, [setResult]);
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">Price optimization</div><h1>Price <em>Optimizer</em></h1><p className="hero-sub">Find the profit-maximizing price within your margin and price-change limits.</p></div>
        <button className="button-primary" onClick={() => void run()} disabled={busy || !sku}>{busy ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />} {busy ? "Calculating..." : "Run optimizer"}</button>
      </div>
      <div className="pricing-layout">
        <section className="panel parameter-panel">
          <PanelTitle eyebrow="DECISION BOUNDS" title="Optimization controls" action={<button className="icon-button" onClick={() => setSkuPickerOpen(!skuPickerOpen)} aria-label="Search or select SKU" aria-expanded={skuPickerOpen}><Search size={15} /></button>} />
          {skuPickerOpen && (
            <div className="price-sku-picker">
              <label className="search-field"><Search size={14} /><input value={skuSearch} onChange={(event) => setSkuSearch(event.target.value)} placeholder="Search SKU code" autoFocus /></label>
              <div className="price-sku-options">
                {filteredSkuOptions.slice(0, skuOptionLimit).map((item) => (
                  <button key={item.sku_id} className={item.sku_id === sku?.sku_id ? "active" : ""} onClick={() => { onSelectSku(item.sku_id); setSkuPickerOpen(false); setSkuSearch(""); }}>
                    <b>{item.sku_id}</b><span>{money(item.unit_price)}</span>
                  </button>
                ))}
              </div>
              {filteredSkuOptions.length > skuOptionLimit && (
                <button className="show-more-button" onClick={() => setSkuOptionLimit((limit) => limit + 50)}>
                  Show more ({filteredSkuOptions.length - skuOptionLimit} remaining)
                </button>
              )}
            </div>
          )}
          <div className="selected-instrument"><span>ACTIVE INSTRUMENT</span><b>{sku?.sku_id ?? "—"}</b><small>REFERENCE {sku ? money(sku.unit_price) : "—"}</small></div>
          <Parameter label="UNIT COST / COGS" value={params.unit_cost} min={0.1} max={Math.max((sku?.unit_price ?? 30) * 1.4, 1)} step={0.1} display={money(params.unit_cost)} onChange={(value) => update("unit_cost", value)} />
          <Parameter label="COMPETITOR PRICE INDEX" value={params.competitor_index} min={0.65} max={1.4} step={0.01} display={`${params.competitor_index.toFixed(2)}×`} onChange={(value) => update("competitor_index", value)} />
          <Parameter label="MINIMUM MARGIN FLOOR" value={params.min_margin_floor} min={0.05} max={0.6} step={0.01} display={`${(params.min_margin_floor * 100).toFixed(0)}%`} onChange={(value) => update("min_margin_floor", value)} />
          <Parameter label="MAXIMUM PRICE CHANGE" value={params.max_price_delta_pct} min={0.05} max={0.5} step={0.01} display={`±${(params.max_price_delta_pct * 100).toFixed(0)}%`} onChange={(value) => update("max_price_delta_pct", value)} />
          <div className="constraint-note"><Zap size={14} /><span>PRICE GUARDRAILS ACTIVE</span><small>Floor + competitive bounds enforced</small></div>
        </section>
        <section className="panel simulator-panel">
          <PanelTitle eyebrow="PROFIT SURFACE // PRICE × EXPECTED DEMAND" title="Constrained profit curve" suffix={sku ? `· ${sku.sku_id}` : ""} />
          <div className="chart-meta-row profit-chart-meta">
            <span><i className="legend-key key-profit" />NET CONTRIBUTION</span><span><i className="legend-key key-demand" />DEMAND RESPONSE</span><span className="amber">▏ OPTIMAL PRICE CROSSHAIR</span>
          </div>
          <MemoProfitChart result={result} params={params} busy={busy} error={error} />
          <div className="infeasible-key"><span className="hatch-key" /> HATCHED REGION: PRICE BELOW MARGIN FLOOR / OUTSIDE POLICY LIMITS</div>
          <div className="alpha-kpis">
            <div className="alpha-card"><span>OPTIMAL PRICE · P*</span><b className="amber">{result ? money(result.optimal_price) : "—"}</b><small>{result ? `${((result.optimal_price / result.reference_price - 1) * 100).toFixed(1)}% vs reference` : "Awaiting solve"}</small></div>
            <div className="alpha-card"><span>PROJECTED DEMAND · Q(P*)</span><b>{result ? <>{number(result.projected_demand)} <small>units / day</small></> : "—"}</b><small>Elasticity-adjusted estimate</small></div>
            <div className="alpha-card positive"><span>NET PROFIT UPLIFT · ΔΠ</span><b>{result ? `${result.net_profit_uplift >= 0 ? "+" : "−"}${money(Math.abs(result.net_profit_uplift))}` : "—"}</b><small><ArrowUpRight size={12} /> ALPHA VS CURRENT PRICE</small></div>
          </div>
          {error && <div className="error-note">{error}</div>}
          <div className="formula-strip"><span>OBJECTIVE</span><code>arg maxₚ [(P − COGS) · min(Q(P), inventory)]</code><span>SUBJECT TO</span><code>P ≥ COGS · (1 + margin floor)</code></div>
        </section>
      </div>
    </>
  );
}

const MemoPricingView = memo(PricingView);

function ExplainabilityView({
  diagnostics,
  calculating,
  onRefresh,
}: {
  diagnostics: Diagnostics | null;
  calculating: boolean;
  onRefresh: () => void;
}) {
  const features = diagnostics?.feature_importance ?? [];
  const maxGain = Math.max(...features.map((item) => item.gain), 1);
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">Model transparency</div><h1>Model <em>Attribution</em></h1><p className="hero-sub">See which signals drive the demand model and how it compares with the baselines.</p></div>
        <div className="heading-meta">
          {calculating && <CalculatingStatus />}
          <button className="button-secondary" onClick={onRefresh} disabled={calculating}>
            <RefreshCw size={14} /> Refresh benchmarks
          </button>
          <span className="model-badge"><span className="status-dot cyan-dot" />{diagnostics?.model_available ? "TRAINED LIGHTGBM MODEL" : "BASELINE MODE · NO TRAINED ARTIFACT"}</span>
        </div>
      </div>
      <div className="explain-grid">
        <section className="panel attribution-panel">
          <PanelTitle eyebrow="GAIN-BASED FEATURE IMPORTANCE" title="Split-gain attribution" action={<span className="tiny-tag">TOP {features.length} SIGNALS</span>} />
          <div className="attribution-axis"><span>FEATURE SIGNAL</span><span>RELATIVE GAIN →</span></div>
          <div className="feature-bars">
            {features.map((item, index) => (
              <div className="feature-row" key={item.feature}>
                <span className="feature-rank">{String(index + 1).padStart(2, "0")}</span>
                <span className="feature-name">{item.feature}</span>
                <div className="bar-rail"><div style={{ width: `${(item.gain / maxGain) * 100}%` }} className={`gain-bar ${index < 2 ? "gain-primary" : ""}`} /></div>
                <span className="gain-value">{item.gain.toFixed(1)}%</span>
              </div>
            ))}
            {!features.length && <div className="empty-note">Feature importance is unavailable until a trained model artifact is present.</div>}
          </div>
          <div className="attribution-foot"><span>GAIN = REDUCTION IN TRAINING LOSS</span><span>FEATURE IMPORTANCE IS NOT CAUSALITY</span></div>
        </section>
        <section className="panel benchmark-panel">
          <PanelTitle eyebrow="CHRONOLOGICAL HOLDOUT" title="Objective comparison" action={<span className="tiny-tag">{(diagnostics?.benchmarks.length ?? 4)} CANDIDATES</span>} />
          <div className="benchmark-table-wrap">
            <table className="terminal-table benchmark-table">
              <thead><tr><th>MODEL</th><th>OBJECTIVE</th><th>WAPE</th><th>RMSLE</th></tr></thead>
              <tbody>
                {(diagnostics?.benchmarks ?? []).map((row) => (
                  <tr key={row.model} className={row.model.startsWith("LightGBM") ? "selected-model" : ""}>
                    <td><span className="table-model">{row.model.startsWith("LightGBM") && <i />}{row.model}</span></td>
                    <td>{row.objective}</td>
                    <td className="mono">{row.wape === null ? "—" : `${(row.wape * 100).toFixed(1)}%`}</td>
                    <td className="mono">{row.rmsle === null ? "—" : row.rmsle.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="benchmark-note"><CircleHelp size={14} /> WAPE / RMSLE are reported only for the persisted validation artifact. Baseline metrics are computed per selected SKU.</div>
        </section>
      </div>
      <section className="panel math-panel">
        <PanelTitle eyebrow="OBJECTIVE INSPECTOR" title="Mathematical formulation" action={<span className="tiny-tag">TWEEDIE DEVIANCE</span>} />
        <div className="math-content">
          <div className="math-equation">𝔼[Y | X] = exp(f(X))</div>
          <div className="math-details">
            <div><span className="math-label">LOSS</span><code>L(μ, y) = (y^(2−p) / ((1−p)(2−p))) − (yμ^(1−p) / (1−p)) + (μ^(2−p) / (2−p))</code></div>
            <div><span className="math-label">VARIANCE POWER</span><code>p = 1.5 ∈ (1, 2) → compound Poisson–Gamma; explicit mass at zero + continuous positive demand.</code></div>
            <div><span className="math-label">FEATURE SOURCE</span><code>{diagnostics?.model_available ? "evaluate_features.evaluate_feature_importance · booster gain" : features.length ? "Mock preview only · no trained artifact in this session" : "Unavailable · trained artifact not found under models/"}</code></div>
          </div>
        </div>
      </section>
    </>
  );
}

const MemoExplainabilityView = memo(ExplainabilityView);

function ScreenerView({
  skus,
  totalSkus,
  results,
  setResults,
}: {
  skus: Sku[];
  totalSkus: number;
  results: PriceResponse[];
  setResults: (value: PriceResponse[]) => void;
}) {
  const [sortKey, setSortKey] = useState<keyof Sku>("revenue");
  const [descending, setDescending] = useState(true);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [scoredOnly, setScoredOnly] = useState(false);
  const [page, setPage] = useState(0);
  const debouncedSearch = useDebouncedValue(search);
  const didAutoScore = useRef(false);
  const sorted = useMemo(
    () => [...skus].sort((a, b) => {
      const left = a[sortKey];
      const right = b[sortKey];
      const comparison = typeof left === "string"
        ? left.localeCompare(String(right))
        : Number(left) - Number(right);
      return comparison * (descending ? -1 : 1);
    }),
    [skus, sortKey, descending],
  );
  const sort = useCallback((key: keyof Sku) => {
    if (sortKey === key) setDescending(!descending);
    else { setSortKey(key); setDescending(true); }
  }, [sortKey, descending]);
  const rescore = useCallback(async () => {
    setRunning(true);
    setStatus(`SCORING UP TO ${Math.min(skus.length, 500)} ACTIONS`);
    try {
      const outcomes: Array<PriceResponse | undefined> = new Array(Math.min(skus.length, 500));
      let nextIndex = 0;
      let failed = 0;
      const worker = async () => {
        while (nextIndex < outcomes.length) {
          const index = nextIndex++;
          const sku = skus[index];
          try {
            outcomes[index] = await optimizePrice({
          sku_id: sku.sku_id,
          unit_cost: Number(Math.max(sku.unit_price * 0.58, 0.01).toFixed(2)),
          competitor_index: 1,
          min_margin_floor: 0.18,
          max_price_delta_pct: 0.25,
          inventory: Math.ceil(sku.avg_daily_demand * 30),
          elasticity: 1.5,
            });
          } catch {
            failed += 1;
          }
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(8, outcomes.length) }, () => worker()),
      );
      const scored = outcomes.filter((outcome): outcome is PriceResponse => outcome !== undefined);
      setResults(scored);
      setStatus(failed
        ? `${scored.length} ACTIONS SCORED · ${failed} FAILED`
        : `${scored.length} ACTIONS SCORED`);
    } catch (error) {
      setStatus(error instanceof Error ? `BATCH FAILED: ${error.message}` : "BATCH FAILED");
    } finally {
      setRunning(false);
    }
  }, [setResults, skus]);
  useEffect(() => {
    if (skus.length && results.length === 0 && !didAutoScore.current) {
      didAutoScore.current = true;
      void rescore();
    }
  }, [skus, results.length, rescore]);
  const resultMap = useMemo(
    () => new Map(results.map((result) => [result.sku_id, result])),
    [results],
  );
  const filteredRows = useMemo(() => sorted.filter((sku) => {
    const matchesSearch = sku.sku_id.toLowerCase().includes(debouncedSearch.trim().toLowerCase());
    return matchesSearch && (!scoredOnly || resultMap.has(sku.sku_id));
  }), [sorted, debouncedSearch, scoredOnly, resultMap]);
  const pageSize = 50;
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const pageRows = useMemo(
    () => filteredRows.slice(page * pageSize, (page + 1) * pageSize),
    [filteredRows, page],
  );
  useEffect(() => setPage(0), [debouncedSearch, sortKey, descending, scoredOnly]);
  const sortableHeader = useCallback((key: keyof Sku, label: string) => (
    <button className="sort-header" onClick={() => sort(key)}>{label}<ArrowDownUp size={11} className={sortKey === key ? "sort-active" : ""} /></button>
  ), [sort, sortKey]);
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">Batch inference</div><h1>Portfolio <em>Screener</em></h1><p className="hero-sub">Score every SKU for pricing opportunities and export the results.</p></div>
        <div className="action-group">
          <button className="button-secondary" onClick={() => exportPricingCsv(skus, results)}><Download size={14} /> Export CSV</button>
          <button className="button-primary" onClick={() => void rescore()} disabled={running || !skus.length}><Play size={14} /> {running ? "Scoring…" : "Batch re-score"}</button>
        </div>
      </div>
      <section className="metric-strip screener-metrics">
        <Metric label="UNIVERSE SIZE" value={number(skus.length, 0)} tone="cyan" />
        <Metric label="PRICING ACTIONS" value={number(results.length, 0)} change="IN CURRENT BATCH" />
        <Metric label="MEDIAN ZERO RATE" value={`${number((skus.reduce((sum, sku) => sum + sku.zero_sale_rate, 0) / Math.max(skus.length, 1)) * 100, 1)}%`} />
        <Metric label="BATCH STATUS" value={running ? "SCORING" : results.length ? "SCORED" : "READY"} tone={running ? "cyan" : "positive"} />
      </section>
      <section className="panel screener-panel">
        <div className="screener-toolbar">
          <div><div className="eyebrow">PORTFOLIO UNIVERSE</div><div className="panel-heading">Pricing opportunity set <span className="count-chip">{skus.length}</span></div><small>500-action batch · top revenue SKUs of {totalSkus.toLocaleString()}</small></div>
          <div className="screener-toolbar-right"><span className={running ? "amber" : "muted"}>{status || "SORTABLE DATA GRID · ALL SKUS"}</span><button className="icon-button" aria-label={filterOpen ? "Hide screener filters" : "Show screener filters"} aria-expanded={filterOpen} onClick={() => setFilterOpen(!filterOpen)}><SlidersHorizontal size={15} /></button></div>
        </div>
        {filterOpen && (
          <div className="screener-filter-bar">
            <label className="search-field"><Search size={14} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search SKU / instrument ID" /></label>
            <div className="segmented">
              <button className={!scoredOnly ? "active" : ""} onClick={() => setScoredOnly(false)}>All</button>
              <button className={scoredOnly ? "active" : ""} onClick={() => setScoredOnly(true)}>Scored only</button>
            </div>
            <span className="muted">{filteredRows.length} of {skus.length} rows</span>
          </div>
        )}
        <div className="table-scroll">
          <table className="terminal-table screener-table">
            <thead><tr>
              <th>{sortableHeader("sku_id", "INSTRUMENT")}</th>
              <th>{sortableHeader("total_units", "TOTAL UNITS")}</th>
              <th>{sortableHeader("avg_daily_demand", "DAILY DEMAND")}</th>
              <th>{sortableHeader("unit_price", "REF PRICE")}</th>
              <th>{sortableHeader("zero_sale_rate", "ZERO RATE")}</th>
              <th>{sortableHeader("revenue", "GROSS REVENUE")}</th>
              <th>OPTIMAL PRICE</th><th>Δ PROFIT / DAY</th>
            </tr></thead>
            <tbody>{pageRows.map((sku, index) => {
              const result = resultMap.get(sku.sku_id);
              return (
                <tr key={sku.sku_id}>
                  <td><span className="row-index">{String(page * pageSize + index + 1).padStart(3, "0")}</span><b>{sku.sku_id}</b></td>
                  <td className="mono">{number(sku.total_units, 0)}</td>
                  <td className="mono">{number(sku.avg_daily_demand, 2)}</td>
                  <td className="mono">{money(sku.unit_price)}</td>
                  <td><span className="zero-rate-cell"><span style={{ width: `${sku.zero_sale_rate * 100}%` }} />{(sku.zero_sale_rate * 100).toFixed(0)}%</span></td>
                  <td className="mono">{money(sku.revenue, 0)}</td>
                  <td className={`mono ${result ? "amber" : "muted"}`}>{result ? money(result.optimal_price) : "—"}</td>
                  <td className={`mono ${result && result.net_profit_uplift > 0 ? "positive" : "muted"}`}>{result ? `${result.net_profit_uplift >= 0 ? "+" : ""}${money(result.net_profit_uplift)}` : "—"}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
        <div className="screener-footer">
          <span>{status || "DISPLAYING PRIORITIZED PORTFOLIO ACTIONS"}</span>
          <span className="list-pager">
            ROWS {filteredRows.length}/{skus.length} · PAGE {page + 1}/{pageCount}
            <button aria-label="Previous screener page" disabled={page === 0} onClick={() => setPage((value) => value - 1)}>‹</button>
            <button aria-label="Next screener page" disabled={page + 1 >= pageCount} onClick={() => setPage((value) => value + 1)}>›</button>
          </span>
        </div>
      </section>
    </>
  );
}

const MemoScreenerView = memo(ScreenerView);

export default function App() {
  const [activeView, setActiveView] = useState<Workspace>("overview");
  const [payload, setPayload] = useState<SkuResponse | null>(null);
  const [selectedSku, setSelectedSku] = useState("");
  const [horizon, setHorizon] = useState(14);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [diagnosticsRefreshEpoch, setDiagnosticsRefreshEpoch] = useState(0);
  const [screenerPayload, setScreenerPayload] = useState<ScreenerResponse | null>(null);
  const [priceResult, setPriceResult] = useState<PriceResponse | null>(null);
  const [batchResults, setBatchResults] = useState<PriceResponse[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [apiError, setApiError] = useState("");
  const [retrying, setRetrying] = useState(false);
  const [retryEpoch, setRetryEpoch] = useState(0);
  const [theme, setTheme] = useState<"dark" | "light">(() =>
    localStorage.getItem("smart-retail-theme") === "light" ? "light" : "dark",
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("smart-retail-theme", theme);
  }, [theme]);

  useEffect(() => {
    let active = true;
    let connecting = false;
    const loadLiveData = async () => {
      if (connecting) return;
      connecting = true;
      setRetrying(true);
      try {
        const live = await connectToLiveApi();
        if (!active) return;
        setPayload((current) =>
          current?.mode === "live_model" && live.portfolio.mode !== "live_model"
            ? current
            : live.portfolio,
        );
        setSelectedSku((current) =>
          live.portfolio.skus.some((sku) => sku.sku_id === current)
            ? current
            : live.portfolio.skus[0]?.sku_id ?? "",
        );
        setDiagnostics(live.diagnostics);
        setApiOnline(true);
        setApiError("");
      } catch (error) {
        if (!active) return;
        setApiOnline(false);
        setApiError(error instanceof Error ? error.message : "Live API connection failed.");
      } finally {
        connecting = false;
        if (active) setRetrying(false);
      }
    };

    void loadLiveData();
    const recoveryTimer = window.setInterval(() => {
      if (connecting || connectionState()) return;
      void checkHealth().then((healthy) => {
        if (active && healthy) void loadLiveData();
      });
    }, 2000);
    return () => {
      active = false;
      window.clearInterval(recoveryTimer);
    };
  }, [retryEpoch]);

  useEffect(() => {
    if (activeView !== "screener" || !apiOnline || screenerPayload) return;
    let active = true;
    void fetchScreener().then((data) => {
      if (!active) return;
      setScreenerPayload(data);
      setApiOnline(connectionState());
    }).catch((error: unknown) => {
      if (!active) return;
      setApiOnline(connectionState());
      setApiError(error instanceof Error ? error.message : "Portfolio Screener request failed.");
    });
    return () => {
      active = false;
    };
  }, [activeView, apiOnline, screenerPayload]);

  useEffect(() => {
    if (!selectedSku || !apiOnline) return;
    let active = true;
    const startedAt = performance.now();
    setForecast(null);
    setForecastLoading(true);
    void (async () => {
      try {
        const data = await fetchForecast(selectedSku, horizon);
        await waitForCalculationFeedback(startedAt);
        if (!active) return;
        setForecast(data);
        setApiOnline(connectionState());
        setApiError("");
      } catch (error: unknown) {
        await waitForCalculationFeedback(startedAt);
        if (!active) return;
        setApiOnline(connectionState());
        setApiError(error instanceof Error ? error.message : "Forecast request failed.");
      } finally {
        if (active) setForecastLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [selectedSku, horizon, apiOnline]);

  useEffect(() => {
    if (activeView !== "explainability" || !apiOnline) return;
    let active = true;
    const startedAt = performance.now();
    setDiagnosticsLoading(true);
    void (async () => {
      try {
        const nextDiagnostics = await fetchDiagnostics();
        await waitForCalculationFeedback(startedAt);
        if (!active) return;
        setDiagnostics(nextDiagnostics);
        setApiError("");
      } catch (error: unknown) {
        await waitForCalculationFeedback(startedAt);
        if (!active) return;
        setApiOnline(connectionState());
        setApiError(error instanceof Error ? error.message : "Model Attribution refresh failed.");
      } finally {
        if (active) setDiagnosticsLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [activeView, apiOnline, diagnosticsRefreshEpoch]);

  const selected = useMemo(
    () => payload?.skus.find((sku) => sku.sku_id === selectedSku),
    [payload, selectedSku],
  );
  const active = views.find((view) => view.id === activeView);
  const liveMode = payload?.mode === "live_model";
  const retryLiveApi = useCallback(() => setRetryEpoch((value) => value + 1), []);
  const refreshDiagnostics = useCallback(() => setDiagnosticsRefreshEpoch((value) => value + 1), []);
  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-glyph"><BarChart3 size={21} strokeWidth={2.4} /></div>
          <div className="brand-text"><span className="brand-name">Smart Retail<br />Intelligence</span></div>
        </div>
        <div className="sidebar-section-label">WORKSPACE</div>
        <nav className="workspace-nav">
          {views.map((view) => {
            const Icon = view.icon;
            return (
              <button key={view.id} className={`nav-item ${activeView === view.id ? "active" : ""}`} onClick={() => setActiveView(view.id)} title={view.label}>
                <Icon size={17} /><span className="nav-label">{view.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-section-label">MODEL STACK</div>
          <div className="stack-item"><span className="stack-dot violet" /><span><b>Demand Model</b><small>{payload?.portfolio.model_available ? "Tweedie LightGBM" : "Baseline Model"}</small></span></div>
          <div className="stack-item"><span className="stack-dot blue" /><span><b>Price Engine</b><small>Log-Log Elasticity</small></span></div>
          <div className="stack-item"><span className={`stack-dot ${apiOnline ? "ok" : "warn"}`} /><span><b>API Bridge</b><small>{apiOnline ? "FastAPI connected" : retrying ? "Connecting to live API" : "Live API unavailable"}</small></span></div>
          <div className="sidebar-user">
            <span className="user-avatar"><Layers size={17} /></span>
            <span><b>Smart-Retail Ops</b><small>Quant Desk · 01</small><em className="online-note"><i />System online</em></span>
          </div>
        </div>
      </aside>
      <div className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            <button className="icon-button" onClick={() => setSidebarCollapsed(!sidebarCollapsed)} aria-label="Toggle navigation">
              {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            </button>
            <LayoutDashboard size={14} /><span>Workspace</span><span className="crumb-sep">/</span><b>{active?.label}</b>
          </div>
          <div className="topbar-right">
            <span className="pill"><CalendarDays size={14} />{new Date().toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" })}</span>
            <span className={`pill connection ${apiOnline ? "is-live" : "is-demo"}`}><span className="status-dot" />{apiOnline ? liveMode ? "Live Model" : "Baseline Mode" : retrying ? "Connecting…" : "API unavailable"}</span>
            <button className="icon-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} title={theme === "dark" ? "Light mode" : "Dark mode"}>
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
            <button className="icon-button" aria-label="Notifications"><Bell size={17} /></button>
          </div>
        </header>
        <div className="view-wrap">
          {payload ? (
            <div className="view-content">
              {apiError && (
                <div className="api-status-banner" role="alert">
                  <span>{apiError}</span>
                  <button className="button-secondary" onClick={retryLiveApi} disabled={retrying}>
                    {retrying ? "Connecting…" : "Retry Live API"}
                  </button>
                </div>
              )}
              {activeView === "overview" && <MemoOverview portfolio={payload.portfolio} metrics={payload.metrics} metricsSource={payload.metrics_source} sku={selected} horizon={horizon} setHorizon={setHorizon} forecast={forecast} forecastLoading={forecastLoading} onSelectSku={setSelectedSku} skus={payload.skus} />}
              {activeView === "pricing" && <MemoPricingView sku={selected} skus={payload.skus} onSelectSku={setSelectedSku} result={priceResult} setResult={setPriceResult} />}
              {activeView === "explainability" && <MemoExplainabilityView diagnostics={diagnostics} calculating={diagnosticsLoading} onRefresh={refreshDiagnostics} />}
              {activeView === "screener" && (
                screenerPayload
                  ? <MemoScreenerView skus={screenerPayload.skus} totalSkus={screenerPayload.total_skus} results={batchResults} setResults={setBatchResults} />
                  : <div className="loading-screen">{retrying ? "Connecting to the Portfolio Screener" : "Loading Portfolio Screener"}</div>
              )}
            </div>
          ) : (
            <div className="loading-screen">
              {apiError ? (
                <div className="api-connect-panel">
                  <span className="error-note">{apiError}</span>
                  <button className="button-primary" onClick={retryLiveApi} disabled={retrying}>
                    {retrying ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />}
                    {retrying ? "Connecting…" : "Retry Live API"}
                  </button>
                </div>
              ) : <><LoaderCircle className="spin" /><span>Connecting to the Smart-Retail data service</span></>}
            </div>
          )}
        </div>
        <footer className="statusbar">
          <span><i className={apiOnline ? "green-bg" : "amber-bg"} />{apiOnline ? "Connected" : retrying ? "Connecting" : "API unavailable"}</span>
          <span>Smart Retail Intelligence <b>1.0.0</b></span>
          <span>Demand Intelligence · {new Date().getFullYear()}</span>
          <span className="status-shortcut"><Activity size={13} /> System status: <b className="ok-text">Nominal</b></span>
        </footer>
      </div>
    </div>
  );
}
