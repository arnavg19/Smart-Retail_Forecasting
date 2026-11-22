import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent, ReactNode } from "react";
import {
  Activity,
  ArrowDownUp,
  ArrowUpRight,
  BarChart3,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Command,
  Download,
  ListFilter,
  LoaderCircle,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Search,
  Settings2,
  SlidersHorizontal,
  Terminal,
  Zap,
} from "lucide-react";
import {
  connectionState,
  exportPricingCsv,
  fetchDiagnostics,
  fetchForecast,
  fetchSkus,
  optimizePrice,
} from "./api";
import type {
  Diagnostics,
  ForecastResponse,
  PriceInput,
  PriceResponse,
  Sku,
  SkuResponse,
} from "./types";

type Workspace = "overview" | "pricing" | "explainability" | "screener";

const views: { id: Workspace; number: string; label: string; icon: typeof Activity }[] = [
  { id: "overview", number: "01", label: "Forecast telemetry", icon: Activity },
  { id: "pricing", number: "02", label: "Price optimizer", icon: SlidersHorizontal },
  { id: "explainability", number: "03", label: "Model attribution", icon: BarChart3 },
  { id: "screener", number: "04", label: "Portfolio screener", icon: ListFilter },
];

const initialPricing: Omit<PriceInput, "sku_id"> = {
  unit_cost: 9.4,
  competitor_index: 1,
  min_margin_floor: 0.18,
  max_price_delta_pct: 0.25,
};

const money = (value: number, decimals = 2) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);

const number = (value: number, decimals = 1) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: decimals }).format(value);

function Header({ online }: { online: boolean }) {
  return (
    <header className="topbar">
      <div className="brand-lockup">
        <div className="brand-glyph"><Command size={17} strokeWidth={2.2} /></div>
        <span className="brand-name">VECTRA</span>
        <span className="brand-divider" />
        <span className="brand-product">RETAIL QUANT SYSTEM</span>
      </div>
      <div className="topbar-right">
        <span className="system-time">{new Date().toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" })}</span>
        <span className={`connection ${online ? "is-live" : "is-demo"}`}>
          <span className="status-dot" />{online ? "API CONNECTED" : "SIMULATION MODE"}
        </span>
        <button className="icon-button" aria-label="System settings"><Settings2 size={16} /></button>
      </div>
    </header>
  );
}

function Metric({
  label,
  value,
  change,
  tone = "neutral",
}: {
  label: string;
  value: string;
  change?: string;
  tone?: "positive" | "cyan" | "neutral";
}) {
  return (
    <div className="metric-cell">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${tone}`}>{value}</div>
      {change && <div className="metric-change"><ArrowUpRight size={12} />{change}</div>}
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
  showOls,
  showPoisson,
}: {
  response: ForecastResponse | null;
  showOls: boolean;
  showPoisson: boolean;
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  if (!response) return <div className="chart-empty"><LoaderCircle className="spin" /> Loading forecast series</div>;
  const points = response.points;
  const historical = points.filter((point) => point.actual != null);
  const future = points.filter((point) => point.prediction != null);
  const values = points.flatMap((point) => [
    point.actual ?? point.prediction ?? 0,
    point.p10 ?? point.actual ?? 0,
    point.p90 ?? point.actual ?? 0,
    ...(showOls && point.ols != null ? [point.ols] : []),
    ...(showPoisson && point.poisson != null ? [point.poisson] : []),
  ]);
  const maxY = Math.max(...values, 1) * 1.16;
  const minY = 0;
  const width = 1000;
  const height = 330;
  const pad = { left: 54, right: 20, top: 20, bottom: 34 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const x = (index: number) => pad.left + (index / Math.max(points.length - 1, 1)) * plotW;
  const y = (value: number) => pad.top + (1 - (value - minY) / (maxY - minY || 1)) * plotH;
  const linePath = (series: Array<number | null | undefined>, offset = 0) =>
    series.reduce((path, value, index) => {
      if (value === null || value === undefined) return path;
      return `${path}${path ? "L" : "M"}${x(index + offset).toFixed(1)},${y(value).toFixed(1)} `;
    }, "");
  const futureOffset = historical.length;
  const forecastPath = linePath(future.map((point) => point.prediction), futureOffset);
  const highPath = linePath(future.map((point) => point.p90), futureOffset);
  const bandPath = highPath
    ? `${highPath} ${future.map((point, index) => `L${x(futureOffset + future.length - index - 1).toFixed(1)},${y(point.p10 ?? 0).toFixed(1)}`).join(" ")} Z`
    : "";
  const historicalPath = linePath(historical.map((point) => point.actual));
  const active = hoverIndex === null ? null : points[hoverIndex];
  const onMove = (event: MouseEvent<SVGSVGElement>) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box) return;
    const relative = ((event.clientX - box.left) / box.width) * width;
    const fraction = Math.max(0, Math.min(1, (relative - pad.left) / plotW));
    setHoverIndex(Math.round(fraction * (points.length - 1)));
  };
  const yTicks = Array.from({ length: 5 }, (_, index) => (maxY * index) / 4);
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
            <stop offset="0%" stopColor="#06B6D4" stopOpacity=".16" />
            <stop offset="100%" stopColor="#06B6D4" stopOpacity=".015" />
          </linearGradient>
        </defs>
        {yTicks.map((tick, index) => (
          <g key={index}>
            <line x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
            <text x={pad.left - 10} y={y(tick) + 4} textAnchor="end" className="axis-label">{number(tick, 0)}</text>
          </g>
        ))}
        <line x1={x(futureOffset)} x2={x(futureOffset)} y1={pad.top} y2={height - pad.bottom} className="forecast-divider" />
        {bandPath && <path d={bandPath} fill="url(#confidence-band)" />}
        {showOls && <path d={linePath(points.map((point) => point.ols))} className="line-ols" />}
        {showPoisson && <path d={linePath(points.map((point) => point.poisson))} className="line-poisson" />}
        <path d={historicalPath} className="line-history" />
        <path d={forecastPath} className="line-forecast" />
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
        {showOls && <span><i className="legend-key key-ols" />OLS</span>}
        {showPoisson && <span><i className="legend-key key-poisson" />POISSON</span>}
        <span><i className="legend-key key-zero" />ZERO-SALE DAY</span>
      </div>
    </div>
  );
}

function Overview({
  portfolio,
  metrics,
  metricsSource,
  sku,
  horizon,
  setHorizon,
  forecast,
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
  onSelectSku: (id: string) => void;
  skus: Sku[];
}) {
  const [search, setSearch] = useState("");
  const [showOls, setShowOls] = useState(false);
  const [showPoisson, setShowPoisson] = useState(false);
  const [watchlistOpen, setWatchlistOpen] = useState(true);
  const filtered = skus.filter((item) => item.sku_id.toLowerCase().includes(search.toLowerCase()));
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">01 // FORECAST &amp; ZERO-INFLATION TELEMETRY</div><h1>Command overview</h1></div>
        <div className="heading-meta"><span className="live-tag"><i />PORTFOLIO LIVE</span><span className="mono">UTC−05:00</span></div>
      </div>
      <section className="metric-strip">
        <Metric label="PORTFOLIO SKUs" value={number(portfolio.total_skus, 0)} change="ACTIVE UNIVERSE" />
        <Metric label="DEMAND / DAY" value={number(portfolio.avg_daily_demand, 0)} tone="cyan" />
        <Metric label={portfolio.model_available ? "WAPE · VALIDATION" : "WAPE · BASELINE"} value={metrics.wape === null ? "—" : `${(metrics.wape * 100).toFixed(1)}%`} change={`${metrics.rmsle === null ? "RMSLE —" : `RMSLE ${metrics.rmsle.toFixed(3)}`} · ${metricsSource}`} tone={metrics.wape === null ? "neutral" : "positive"} />
        <Metric label="ZERO-SALE FREQUENCY" value={`${number(portfolio.zero_sale_rate * 100, 1)}%`} tone="neutral" />
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
                {filtered.map((item, index) => (
                  <button key={item.sku_id} className={`watch-item ${sku?.sku_id === item.sku_id ? "selected" : ""}`} onClick={() => onSelectSku(item.sku_id)}>
                    <span className="watch-symbol"><b>{item.sku_id}</b><small>{index < 3 ? "HIGH VELOCITY" : "CORE SKU"}</small></span>
                    <span className="watch-demand"><b>{number(item.avg_daily_demand, 1)}</b><small>{(item.zero_sale_rate * 100).toFixed(0)}% zeros</small></span>
                  </button>
                ))}
                {filtered.length === 0 && <div className="empty-note">No matching instruments.</div>}
              </div>
              <div className="watchlist-footer"><span><i className="status-dot" />{filtered.length} INSTRUMENTS</span><span>UNIVERSE A—Z</span></div>
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
              <button className={`toggle-chip ${showOls ? "active" : ""}`} onClick={() => setShowOls(!showOls)}><i className="line-dot ols-dot" />OLS</button>
              <button className={`toggle-chip ${showPoisson ? "active" : ""}`} onClick={() => setShowPoisson(!showPoisson)}><i className="line-dot poisson-dot" />POISSON</button>
            </div>
          </div>
          <div className="chart-meta-row">
            <span className="chart-model"><span className="status-dot cyan-dot" />{forecast?.model ?? "Connecting to forecast service"}</span>
            <span>ZERO DAYS: <b className="crimson">{number((sku?.zero_sale_rate ?? 0) * 60, 0)} / 60</b></span>
            <span>FORECAST: <b className="cyan">{horizon} DAY HORIZON</b></span>
          </div>
          <ForecastChart response={forecast} showOls={showOls} showPoisson={showPoisson} />
          <div className="forecast-foot">
            <span><span className="foot-key">MODEL</span> {forecast?.model_available ? "LIGHTGBM TWEEDIE · p=1.5" : "LIVE-DATA FALLBACK"}</span>
            <span><span className="foot-key">SOURCE</span> {portfolio.data_source}</span>
            <span>ZERO SALES ARE OBSERVED SIGNALS, NOT MISSING DATA</span>
          </div>
        </section>
      </div>
    </>
  );
}

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

function ProfitChart({ result, params }: { result: PriceResponse | null; params: Omit<PriceInput, "sku_id"> }) {
  if (!result || result.profit_curve.length === 0) return <div className="chart-empty"><LoaderCircle className="spin" /> Calculating constrained optimum</div>;
  const width = 920;
  const height = 300;
  const pad = { left: 54, right: 26, top: 20, bottom: 35 };
  const values = result.profit_curve;
  const xMin = result.reference_price * (1 - params.max_price_delta_pct);
  const xMax = result.reference_price * (1 + params.max_price_delta_pct);
  const lowerFeasible = Math.max(
    result.reference_price * (1 - params.max_price_delta_pct),
    params.unit_cost * (1 + params.min_margin_floor),
  );
  const maxProfit = Math.max(...values.map((point) => point.profit), 1);
  const x = (price: number) => pad.left + ((price - xMin) / (xMax - xMin || 1)) * (width - pad.left - pad.right);
  const y = (profit: number) => pad.top + (1 - profit / (maxProfit * 1.15)) * (height - pad.top - pad.bottom);
  const profitPath = values.map((point, index) => `${index ? "L" : "M"}${x(point.price).toFixed(1)},${y(point.profit).toFixed(1)}`).join(" ");
  const demandMax = Math.max(...values.map((point) => point.demand), 1);
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

function PricingView({
  sku,
  result,
  setResult,
}: {
  sku: Sku | undefined;
  result: PriceResponse | null;
  setResult: (value: PriceResponse | null) => void;
}) {
  const [params, setParams] = useState(initialPricing);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!sku) return;
    setParams({ ...initialPricing, unit_cost: Math.max(sku.unit_price * 0.58, 0.1) });
  }, [sku?.sku_id, sku?.unit_price]);
  const run = async () => {
    if (!sku) return;
    const reference = sku.unit_price * params.competitor_index;
    if (params.unit_cost * (1 + params.min_margin_floor) > reference * (1 + params.max_price_delta_pct)) {
      setError("Margin floor is outside the permitted price-change interval. Adjust the cost, margin, or bound.");
      setResult(null);
      return;
    }
    setBusy(true);
    setError("");
    try {
      setResult(await optimizePrice({ sku_id: sku.sku_id, ...params }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Price optimization failed.");
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => {
    const timer = window.setTimeout(() => void run(), 180);
    return () => window.clearTimeout(timer);
  }, [sku?.sku_id, params.unit_cost, params.competitor_index, params.min_margin_floor, params.max_price_delta_pct]);
  const update = (key: keyof typeof params, value: number) =>
    setParams((current) => ({ ...current, [key]: value }));
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">02 // CONSTRAINED LOG-LOG ELASTICITY &amp; MARGIN OPTIMIZER</div><h1>Dynamic pricing simulator</h1></div>
        <button className="button-primary" onClick={() => void run()} disabled={busy || !sku}>{busy ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />} {busy ? "SOLVING" : "RUN OPTIMIZER"}</button>
      </div>
      <div className="pricing-layout">
        <section className="panel parameter-panel">
          <PanelTitle eyebrow="DECISION BOUNDS" title="Optimization controls" action={<Settings2 size={15} className="muted-icon" />} />
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
          <ProfitChart result={result} params={params} />
          <div className="infeasible-key"><span className="hatch-key" /> HATCHED REGION: PRICE BELOW MARGIN FLOOR / OUTSIDE POLICY LIMITS</div>
          <div className="alpha-kpis">
            <div className="alpha-card"><span>OPTIMAL PRICE · P*</span><b className="amber">{result ? money(result.optimal_price) : "—"}</b><small>{result ? `${((result.optimal_price / result.reference_price - 1) * 100).toFixed(1)}% vs reference` : "Awaiting solve"}</small></div>
            <div className="alpha-card"><span>PROJECTED DEMAND · Q(P*)</span><b>{result ? `${number(result.projected_demand)} <small>UNITS / DAY</small>` : "—"}</b><small>Elasticity-adjusted estimate</small></div>
            <div className="alpha-card positive"><span>NET PROFIT UPLIFT · ΔΠ</span><b>{result ? `${result.net_profit_uplift >= 0 ? "+" : "−"}${money(Math.abs(result.net_profit_uplift))}` : "—"}</b><small><ArrowUpRight size={12} /> ALPHA VS CURRENT PRICE</small></div>
          </div>
          {error && <div className="error-note">{error}</div>}
          <div className="formula-strip"><span>OBJECTIVE</span><code>arg maxₚ [(P − COGS) · min(Q(P), inventory)]</code><span>SUBJECT TO</span><code>P ≥ COGS · (1 + margin floor)</code></div>
        </section>
      </div>
    </>
  );
}

function ExplainabilityView({ diagnostics }: { diagnostics: Diagnostics | null }) {
  const features = diagnostics?.feature_importance ?? [];
  const maxGain = Math.max(...features.map((item) => item.gain), 1);
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">03 // SPLIT-GAIN ATTRIBUTION &amp; BENCHMARK MATRIX</div><h1>Model explainability</h1></div>
        <span className="model-badge"><span className="status-dot cyan-dot" />{diagnostics?.model_available ? "TRAINED LIGHTGBM MODEL" : features.length ? "SIMULATED ATTRIBUTION" : "NO TRAINED ARTIFACT"}</span>
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
          <PanelTitle eyebrow="CHRONOLOGICAL HOLDOUT" title="Objective comparison" action={<span className="tiny-tag">4 CANDIDATES</span>} />
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

function ScreenerView({
  skus,
  results,
  setResults,
}: {
  skus: Sku[];
  results: PriceResponse[];
  setResults: (value: PriceResponse[]) => void;
}) {
  const [sortKey, setSortKey] = useState<keyof Sku>("revenue");
  const [descending, setDescending] = useState(true);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");
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
  const sort = (key: keyof Sku) => {
    if (sortKey === key) setDescending(!descending);
    else { setSortKey(key); setDescending(true); }
  };
  const rescore = async () => {
    setRunning(true);
    setStatus("SCORING PORTFOLIO");
    try {
      const scored = await Promise.all(skus.slice(0, 50).map((sku) =>
        optimizePrice({
          sku_id: sku.sku_id,
          unit_cost: Math.max(sku.unit_price * 0.58, 0.01),
          competitor_index: 1,
          min_margin_floor: 0.18,
          max_price_delta_pct: 0.25,
        }),
      ));
      setResults(scored);
      setStatus(`${scored.length} SKUS RE-SCORED`);
    } catch (error) {
      setStatus(error instanceof Error ? `BATCH FAILED: ${error.message}` : "BATCH FAILED");
    } finally {
      setRunning(false);
    }
  };
  const resultMap = new Map(results.map((result) => [result.sku_id, result]));
  const sortableHeader = (key: keyof Sku, label: string) => (
    <button className="sort-header" onClick={() => sort(key)}>{label}<ArrowDownUp size={11} className={sortKey === key ? "sort-active" : ""} /></button>
  );
  return (
    <>
      <div className="view-heading">
        <div><div className="eyebrow">04 // LIVE INFERENCE &amp; SKU PORTFOLIO SCREENER</div><h1>Batch inference &amp; screener</h1></div>
        <div className="action-group">
          <button className="button-secondary" onClick={() => exportPricingCsv(skus, results)}><Download size={14} /> EXPORT CSV</button>
          <button className="button-primary" onClick={() => void rescore()} disabled={running}><Play size={14} /> {running ? "SCORING…" : "BATCH RE-SCORE"}</button>
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
          <div><div className="eyebrow">PORTFOLIO UNIVERSE</div><div className="panel-heading">Pricing opportunity set <span className="count-chip">{skus.length}</span></div></div>
          <div className="screener-toolbar-right"><span className={running ? "amber" : "muted"}>{status || "SORTABLE DATA GRID · TOP 500 SKUS"}</span><button className="icon-button" aria-label="Filter results"><SlidersHorizontal size={15} /></button></div>
        </div>
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
            <tbody>{sorted.map((sku, index) => {
              const result = resultMap.get(sku.sku_id);
              return (
                <tr key={sku.sku_id}>
                  <td><span className="row-index">{String(index + 1).padStart(3, "0")}</span><b>{sku.sku_id}</b></td>
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
        <div className="screener-footer"><span>{status || "DISPLAYING PORTFOLIO INSTRUMENTS"}</span><span>ROWS {skus.length} · RESULTS SORTED BY {String(sortKey).toUpperCase()}</span></div>
      </section>
    </>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState<Workspace>("overview");
  const [payload, setPayload] = useState<SkuResponse | null>(null);
  const [selectedSku, setSelectedSku] = useState("");
  const [horizon, setHorizon] = useState(14);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [priceResult, setPriceResult] = useState<PriceResponse | null>(null);
  const [batchResults, setBatchResults] = useState<PriceResponse[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    void fetchSkus().then((data) => {
      setPayload(data);
      setSelectedSku(data.skus[0]?.sku_id ?? "");
      setApiOnline(connectionState());
    }).catch((error: unknown) => setApiError(error instanceof Error ? error.message : "Portfolio request failed."));
    void fetchDiagnostics().then((data) => {
      setDiagnostics(data);
      setApiOnline(connectionState());
    }).catch((error: unknown) => setApiError(error instanceof Error ? error.message : "Diagnostics request failed."));
  }, []);
  useEffect(() => {
    if (!selectedSku) return;
    void fetchForecast(selectedSku, horizon).then((data) => {
      setForecast(data);
      setApiOnline(connectionState());
    }).catch((error: unknown) => setApiError(error instanceof Error ? error.message : "Forecast request failed."));
  }, [selectedSku, horizon]);
  const selected = payload?.skus.find((sku) => sku.sku_id === selectedSku);
  const active = views.find((view) => view.id === activeView);
  const ActiveIcon = active?.icon ?? Activity;
  return (
    <div className="app-shell">
      <Header online={apiOnline} />
      <div className="app-body">
        <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
          <button className="sidebar-collapse icon-button" onClick={() => setSidebarCollapsed(!sidebarCollapsed)} aria-label="Toggle navigation">
            {sidebarCollapsed ? <ChevronRight size={15} /> : <ChevronLeft size={15} />}
          </button>
          <div className="sidebar-section-label">{sidebarCollapsed ? "WS" : "WORKSPACES"}</div>
          <nav className="workspace-nav">
            {views.map((view) => {
              const Icon = view.icon;
              return <button key={view.id} className={`nav-item ${activeView === view.id ? "active" : ""}`} onClick={() => setActiveView(view.id)} title={sidebarCollapsed ? view.label : undefined}>
                <Icon size={16} /><span className="nav-number">{view.number}</span><span className="nav-label">{view.label}</span>{activeView === view.id && <span className="nav-active-mark" />}
              </button>;
            })}
          </nav>
          {!sidebarCollapsed && (
            <>
              <div className="sidebar-bottom">
                <div className="sidebar-section-label">MODEL STACK</div>
                <div className="stack-item"><span className={`stack-status ${payload?.portfolio.model_available ? "green" : "amber-bg"}`} /><span><b>DEMAND MODEL</b><small>{payload?.portfolio.model_available ? "TWEEDIE · p=1.5" : "ARTIFACT NOT FOUND"}</small></span></div>
                <div className="stack-item"><span className="stack-status green" /><span><b>PRICE ENGINE</b><small>LOG-LOG ELASTICITY</small></span></div>
                <div className="stack-item"><span className={`stack-status ${apiOnline ? "green" : "amber-bg"}`} /><span><b>API BRIDGE</b><small>{apiOnline ? "FASTAPI · CONNECTED" : "DEMO DATA ACTIVE"}</small></span></div>
              </div>
              <div className="sidebar-user"><div className="user-avatar">V</div><span><b>VECTRA OPS</b><small>QUANT DESK · 01</small></span><ChevronDown size={13} /></div>
            </>
          )}
        </aside>
        <main className="main-content">
          <div className="breadcrumb"><span>VECTRA</span><ChevronRight size={12} /><span>WORKSPACE</span><ChevronRight size={12} /><b><ActiveIcon size={12} />{active?.number} // {active?.label.toUpperCase()}</b></div>
          {payload ? (
            <div className="view-content">
              {apiError && <div className="error-note">{apiError}</div>}
              {activeView === "overview" && <Overview portfolio={payload.portfolio} metrics={payload.metrics} metricsSource={payload.metrics_source} sku={selected} horizon={horizon} setHorizon={setHorizon} forecast={forecast} onSelectSku={setSelectedSku} skus={payload.skus} />}
              {activeView === "pricing" && <PricingView sku={selected} result={priceResult} setResult={setPriceResult} />}
              {activeView === "explainability" && <ExplainabilityView diagnostics={diagnostics} />}
              {activeView === "screener" && <ScreenerView skus={payload.skus} results={batchResults} setResults={setBatchResults} />}
            </div>
          ) : (
            <div className="loading-screen">{apiError ? <span className="error-note">{apiError}</span> : <><LoaderCircle className="spin" /><span>CONNECTING TO VECTRA DATA PLANE</span></>}</div>
          )}
          <footer className="statusbar"><span><i className={apiOnline ? "green-bg" : "amber-bg"} />{apiOnline ? "CONNECTED" : "OFFLINE FALLBACK"}</span><span>VECTRA TERMINAL <b>1.0.0</b></span><span>DEMAND INTELLIGENCE · {new Date().getFullYear()}</span><span className="status-shortcut"><Terminal size={12} /> SYSTEM STATUS: NOMINAL</span></footer>
        </main>
      </div>
    </div>
  );
}
