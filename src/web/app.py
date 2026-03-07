"""Aiohttp web server for the trading dashboard.

Provides:
    - / — Main dashboard (HTML + SSE for real-time updates)
    - /api/status — Full engine status JSON
    - /api/positions — Open positions
    - /api/signals — Current quant signals
    - /api/activity — Recent trade activity
    - /events — Server-Sent Events stream
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from src.core.engine import TradingEngine

logger = logging.getLogger(__name__)


class Dashboard:
    """Web dashboard server backed by aiohttp."""

    def __init__(self, engine: "TradingEngine", port: int = 8090) -> None:
        self._engine = engine
        self._port = port
        self._app = web.Application()
        self._sse_clients: list[web.StreamResponse] = []
        self._setup_routes()

    def _setup_routes(self) -> None:
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/api/status", self._handle_status)
        self._app.router.add_get("/api/positions", self._handle_positions)
        self._app.router.add_get("/api/signals", self._handle_signals)
        self._app.router.add_get("/api/activity", self._handle_activity)
        self._app.router.add_get("/events", self._handle_sse)

    async def start(self) -> None:
        """Start the web server (non-blocking)."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._port)
        await site.start()
        logger.info("Dashboard running at http://0.0.0.0:%d", self._port)

        # Start SSE broadcast loop
        asyncio.create_task(self._sse_broadcast_loop())

    # ── API Handlers ──────────────────────────────────────────

    async def _handle_status(self, request: web.Request) -> web.Response:
        status = self._engine.status()
        return web.json_response(status, dumps=_json_dumps)

    async def _handle_positions(self, request: web.Request) -> web.Response:
        positions = []
        for symbol, pos in self._engine._open_positions.items():
            quote = self._engine._stream.get_option_quote(symbol)
            current_mid = 0.0
            if quote:
                bid = quote.get("bid", 0)
                ask = quote.get("ask", 0)
                current_mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

            entry = float(pos.get("entry_premium", 0))
            pnl_pct = ((current_mid - entry) / entry * 100) if entry > 0 else 0

            positions.append({
                "symbol": symbol,
                "underlying": pos.get("underlying", ""),
                "option_type": pos.get("option_type", ""),
                "strike": str(pos.get("strike", "")),
                "qty": pos.get("qty", 0),
                "entry_premium": entry,
                "current_premium": round(current_mid, 2),
                "pnl_pct": round(pnl_pct, 1),
                "confidence": pos.get("confidence", 0),
                "entry_time": pos.get("entry_time", "").isoformat() if isinstance(pos.get("entry_time"), datetime) else "",
            })
        return web.json_response(positions, dumps=_json_dumps)

    async def _handle_signals(self, request: web.Request) -> web.Response:
        signals = {}

        # VIX
        vix = self._engine._vix.latest
        if vix:
            signals["vix"] = {
                "level": vix.vix_level,
                "regime": vix.regime.value,
                "iv_percentile": vix.iv_percentile,
                "iv_rank": vix.iv_rank,
                "rv_iv_spread": vix.rv_iv_spread,
                "size_multiplier": vix.size_multiplier,
                "should_trade": vix.should_trade,
            }

        # GEX per underlying
        signals["gex"] = {}
        for sym in self._engine._settings.underlying_list:
            gex = self._engine._gex.get_latest(sym)
            if gex:
                signals["gex"][sym] = {
                    "regime": gex.regime.value,
                    "total_gex": gex.total_gex,
                    "call_wall": gex.call_wall,
                    "put_wall": gex.put_wall,
                    "flip_point": gex.flip_point,
                }

        # Flow
        flow = self._engine._flow.latest
        if flow:
            signals["flow"] = {
                "put_call_ratio": flow.put_call_ratio,
                "flow_direction": flow.flow_direction,
                "smart_money_bias": flow.smart_money_bias,
                "extreme_reading": flow.extreme_reading,
                "call_volume": flow.call_volume,
                "put_volume": flow.put_volume,
            }

        # Sentiment
        sent = self._engine._sentiment.latest
        if sent:
            signals["sentiment"] = {
                "composite_score": sent.composite_score,
                "regime": sent.regime.value,
                "fear_greed_index": sent.fear_greed_index,
                "contrarian_signal": sent.contrarian_signal,
                "news_catalyst": sent.news_catalyst,
            }

        # Macro
        macro = self._engine._macro.latest
        if macro:
            signals["macro"] = {
                "is_blackout": macro.is_blackout,
                "reason": macro.blackout_reason,
                "minutes_to_event": macro.minutes_to_event,
                "events_today": len(macro.events_today),
            }

        # Internals
        internals = self._engine._internals.latest
        if internals:
            signals["internals"] = {
                "nyse_tick": internals.nyse_tick,
                "tick_extreme": internals.tick_extreme,
                "ad_ratio": internals.advance_decline_ratio,
                "breadth_score": internals.breadth_score,
                "cumulative_delta": internals.cumulative_delta,
            }

        # OptionsAI per underlying
        signals["optionsai"] = {}
        for sym in self._engine._settings.underlying_list:
            oai = self._engine._optionsai.get_latest(sym)
            if oai:
                signals["optionsai"][sym] = {
                    "iv_skew": oai.iv_skew,
                    "move_amount": oai.move_amount,
                    "move_percent": oai.move_percent,
                    "implied_high": oai.implied_high,
                    "implied_low": oai.implied_low,
                    "call_iv": oai.call_iv,
                    "put_iv": oai.put_iv,
                    "strategy_bias": oai.strategy_bias,
                    "bullish_strategies": oai.bullish_strategies,
                    "bearish_strategies": oai.bearish_strategies,
                    "strategy_names": oai.strategy_names,
                    "price_vs_range": oai.price_vs_implied_range,
                    "earnings_nearby": self._engine._optionsai.has_earnings(sym),
                }

        # Tick momentum per underlying
        signals["momentum"] = {}
        for sym in self._engine._settings.underlying_list:
            mom = self._engine._stream.get_momentum(sym)
            if mom:
                signals["momentum"][sym] = {
                    "price": mom.latest_price,
                    "direction": round(mom.direction, 3),
                    "roc_pct": round(mom.roc_pct, 4),
                    "speed": round(mom.speed, 4),
                }

        return web.json_response(signals, dumps=_json_dumps)

    async def _handle_activity(self, request: web.Request) -> web.Response:
        try:
            trades = self._engine._db.get_trade_history(limit=50)
        except Exception:
            trades = []
        return web.json_response(trades, dumps=_json_dumps)

    # ── SSE (Server-Sent Events) ──────────────────────────────

    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)
        self._sse_clients.append(response)

        try:
            while True:
                await asyncio.sleep(1)
                if response.task is None or response.task.done():
                    break
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._sse_clients.remove(response)

        return response

    async def _sse_broadcast_loop(self) -> None:
        """Push updates to all SSE clients every 2 seconds."""
        while True:
            try:
                if self._sse_clients:
                    data = {
                        "status": self._engine.status(),
                        "prices": {
                            sym: str(p)
                            for sym, p in self._engine._last_prices.items()
                        },
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    payload = f"data: {_json_dumps(data)}\n\n"

                    dead = []
                    for client in self._sse_clients:
                        try:
                            await client.write(payload.encode())
                        except (ConnectionResetError, Exception):
                            dead.append(client)

                    for d in dead:
                        self._sse_clients.remove(d)

            except Exception:
                logger.debug("SSE broadcast error")

            await asyncio.sleep(2)

    # ── HTML Dashboard ────────────────────────────────────────

    async def _handle_index(self, request: web.Request) -> web.Response:
        return web.Response(text=_DASHBOARD_HTML, content_type="text/html")


def _json_dumps(obj: object) -> str:
    """JSON serializer that handles Decimal and datetime."""
    def default(o: object) -> object:
        from decimal import Decimal
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=default)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>0DTE Scalper Dashboard</title>
<style>
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9;
          --green: #3fb950; --red: #f85149; --yellow: #d29922; --blue: #58a6ff; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }
  .header { background: var(--card); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 16px; color: var(--blue); }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .status-dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-dot.off { background: var(--red); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; padding: 12px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 14px; }
  .card h2 { font-size: 12px; text-transform: uppercase; color: #8b949e; margin-bottom: 10px; letter-spacing: 0.5px; }
  .row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d; }
  .row:last-child { border-bottom: none; }
  .label { color: #8b949e; }
  .val { font-weight: 600; }
  .val.green { color: var(--green); }
  .val.red { color: var(--red); }
  .val.yellow { color: var(--yellow); }
  .val.blue { color: var(--blue); }
  .gauge { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .gauge-bar { flex: 1; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }
  .gauge-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: #8b949e; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; }
  .tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; }
  .tag.call { background: rgba(63,185,80,0.15); color: var(--green); }
  .tag.put { background: rgba(248,81,73,0.15); color: var(--red); }
  .tag.crisis { background: rgba(248,81,73,0.15); color: var(--red); }
  .tag.normal { background: rgba(88,166,255,0.15); color: var(--blue); }
  .tag.halted { background: rgba(248,81,73,0.2); color: var(--red); }
  .tag.active { background: rgba(63,185,80,0.15); color: var(--green); }
  #activity { max-height: 300px; overflow-y: auto; }
</style>
</head>
<body>
<div class="header">
  <h1>0DTE Options Scalper</h1>
  <div><span class="status-dot off" id="connDot"></span><span id="connText">Connecting...</span></div>
</div>
<div class="grid">

  <!-- Account -->
  <div class="card">
    <h2>Account</h2>
    <div class="row"><span class="label">Portfolio Value</span><span class="val" id="portfolioValue">--</span></div>
    <div class="row"><span class="label">Daily P&L</span><span class="val" id="dailyPnl">--</span></div>
    <div class="row"><span class="label">Drawdown</span><span class="val" id="drawdown">--</span></div>
    <div class="row"><span class="label">Win Rate</span><span class="val" id="winRate">--</span></div>
    <div class="row"><span class="label">Trades Today</span><span class="val" id="tradesToday">--</span></div>
    <div class="row"><span class="label">Open Positions</span><span class="val blue" id="openPos">--</span></div>
  </div>

  <!-- Prices -->
  <div class="card">
    <h2>Underlyings</h2>
    <div id="pricesPanel">--</div>
  </div>

  <!-- VIX / Volatility -->
  <div class="card">
    <h2>VIX / Volatility</h2>
    <div class="row"><span class="label">VIX</span><span class="val" id="vixLevel">--</span></div>
    <div class="row"><span class="label">Regime</span><span class="val" id="vixRegime">--</span></div>
    <div class="row"><span class="label">IV Percentile</span><span class="val" id="ivPctile">--</span></div>
    <div class="row"><span class="label">Size Multiplier</span><span class="val" id="sizeMult">--</span></div>
  </div>

  <!-- Risk -->
  <div class="card">
    <h2>Risk & Greeks</h2>
    <div class="row"><span class="label">Circuit Breaker</span><span class="val" id="cbStatus">--</span></div>
    <div class="row"><span class="label">PDT Remaining</span><span class="val" id="pdtRemain">--</span></div>
    <div class="gauge"><span class="label" style="width:50px">Delta</span><div class="gauge-bar"><div class="gauge-fill" id="deltaBar" style="width:0; background:var(--blue)"></div></div><span class="val" id="deltaVal">0</span></div>
    <div class="gauge"><span class="label" style="width:50px">Gamma</span><div class="gauge-bar"><div class="gauge-fill" id="gammaBar" style="width:0; background:var(--green)"></div></div><span class="val" id="gammaVal">0</span></div>
    <div class="gauge"><span class="label" style="width:50px">Theta</span><div class="gauge-bar"><div class="gauge-fill" id="thetaBar" style="width:0; background:var(--red)"></div></div><span class="val" id="thetaVal">0</span></div>
    <div class="gauge"><span class="label" style="width:50px">Vega</span><div class="gauge-bar"><div class="gauge-fill" id="vegaBar" style="width:0; background:var(--yellow)"></div></div><span class="val" id="vegaVal">0</span></div>
  </div>

  <!-- Quant Signals -->
  <div class="card">
    <h2>Quant Signals</h2>
    <div class="row"><span class="label">Macro Blackout</span><span class="val" id="macroStatus">--</span></div>
    <div class="row"><span class="label">Flow Direction</span><span class="val" id="flowDir">--</span></div>
    <div class="row"><span class="label">P/C Ratio</span><span class="val" id="pcr">--</span></div>
    <div class="row"><span class="label">Sentiment</span><span class="val" id="sentRegime">--</span></div>
    <div class="row"><span class="label">Breadth</span><span class="val" id="breadth">--</span></div>
  </div>

  <!-- OptionsAI -->
  <div class="card">
    <h2>OptionsAI Signals</h2>
    <div id="optionsaiPanel"><span style="color:#8b949e">Waiting...</span></div>
  </div>

  <!-- Positions -->
  <div class="card" style="grid-column: span 2">
    <h2>Open Positions</h2>
    <table>
      <thead><tr><th>Symbol</th><th>Type</th><th>Strike</th><th>Qty</th><th>Entry</th><th>Current</th><th>P&L</th><th>Conf</th></tr></thead>
      <tbody id="positionsBody"><tr><td colspan="8" style="text-align:center;color:#8b949e">No positions</td></tr></tbody>
    </table>
  </div>

  <!-- Activity -->
  <div class="card" style="grid-column: span 2">
    <h2>Recent Activity</h2>
    <div id="activity">Loading...</div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

// SSE connection
let es;
function connectSSE() {
  es = new EventSource('/events');
  es.onopen = () => {
    $('connDot').className = 'status-dot live';
    $('connText').textContent = 'Connected';
  };
  es.onmessage = e => {
    try { updateDashboard(JSON.parse(e.data)); } catch(err) { console.error(err); }
  };
  es.onerror = () => {
    $('connDot').className = 'status-dot off';
    $('connText').textContent = 'Reconnecting...';
    es.close();
    setTimeout(connectSSE, 3000);
  };
}
connectSSE();

function updateDashboard(data) {
  const s = data.status || {};
  const risk = s.risk || {};
  const greeks = risk.portfolio_greeks || {};
  const pdt = risk.pdt || {};
  const cb = risk.circuit_breaker || {};
  const vix = s.vix || {};

  // Account
  $('portfolioValue').textContent = '$' + Number(risk.portfolio_value || 0).toLocaleString();
  const pnl = Number(risk.daily_pnl || 0);
  $('dailyPnl').textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
  $('dailyPnl').className = 'val ' + (pnl >= 0 ? 'green' : 'red');
  $('drawdown').textContent = risk.drawdown || '--';
  $('winRate').textContent = risk.win_rate || 'N/A';
  $('tradesToday').textContent = risk.trades_today || 0;
  $('openPos').textContent = s.open_positions || 0;

  // Prices
  const prices = data.prices || {};
  let priceHtml = '';
  for (const [sym, p] of Object.entries(prices)) {
    priceHtml += '<div class="row"><span class="label">' + sym + '</span><span class="val blue">$' + Number(p).toFixed(2) + '</span></div>';
  }
  $('pricesPanel').innerHTML = priceHtml || '<span style="color:#8b949e">Waiting for data...</span>';

  // VIX
  $('vixLevel').textContent = vix.level || '--';
  $('vixRegime').innerHTML = vix.regime ? '<span class="tag ' + (vix.regime === 'crisis' ? 'crisis' : 'normal') + '">' + vix.regime + '</span>' : '--';

  // Greeks gauges
  const maxD = 50, maxG = 20, maxT = 100, maxV = 30;
  setGauge('delta', greeks.delta, maxD);
  setGauge('gamma', greeks.gamma, maxG);
  setGauge('theta', Math.abs(greeks.theta || 0), maxT);
  setGauge('vega', greeks.vega, maxV);

  // Risk
  $('cbStatus').innerHTML = cb.halted ? '<span class="tag halted">HALTED</span> ' + (cb.reason || '') : '<span class="tag active">Active</span>';
  $('pdtRemain').textContent = pdt.remaining !== undefined ? pdt.remaining + ' trades' : '--';
}

function setGauge(name, val, max) {
  const pct = Math.min(100, Math.abs(val || 0) / max * 100);
  $(name + 'Bar').style.width = pct + '%';
  $(name + 'Val').textContent = (val || 0).toFixed(1);
}

// Fetch positions every 5s
setInterval(async () => {
  try {
    const r = await fetch('/api/positions');
    const positions = await r.json();
    const tbody = $('positionsBody');
    if (!positions.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#8b949e">No positions</td></tr>';
      return;
    }
    tbody.innerHTML = positions.map(p => {
      const pnlClass = p.pnl_pct >= 0 ? 'green' : 'red';
      const typeClass = p.option_type === 'call' ? 'call' : 'put';
      return '<tr><td>' + p.underlying + '</td><td><span class="tag ' + typeClass + '">' + p.option_type + '</span></td><td>$' + p.strike + '</td><td>' + p.qty + '</td><td>$' + p.entry_premium.toFixed(2) + '</td><td>$' + p.current_premium.toFixed(2) + '</td><td class="' + pnlClass + '">' + (p.pnl_pct >= 0 ? '+' : '') + p.pnl_pct.toFixed(1) + '%</td><td>' + p.confidence + '</td></tr>';
    }).join('');
  } catch(e) {}
}, 5000);

// Fetch signals every 10s
setInterval(async () => {
  try {
    const r = await fetch('/api/signals');
    const sig = await r.json();

    if (sig.vix) {
      $('ivPctile').textContent = sig.vix.iv_percentile + '%';
      $('sizeMult').textContent = sig.vix.size_multiplier + 'x';
    }
    if (sig.macro) {
      $('macroStatus').innerHTML = sig.macro.is_blackout ? '<span class="tag crisis">BLACKOUT</span> ' + sig.macro.reason : '<span class="tag active">Clear</span>';
    }
    if (sig.flow) {
      const fd = sig.flow.flow_direction;
      $('flowDir').textContent = (fd > 0 ? 'Bullish' : fd < 0 ? 'Bearish' : 'Neutral') + ' (' + fd.toFixed(2) + ')';
      $('flowDir').className = 'val ' + (fd > 0 ? 'green' : fd < 0 ? 'red' : '');
      $('pcr').textContent = sig.flow.put_call_ratio;
    }
    if (sig.sentiment) {
      $('sentRegime').innerHTML = '<span class="tag ' + (sig.sentiment.regime.includes('fear') ? 'crisis' : 'normal') + '">' + sig.sentiment.regime + '</span> (F&G: ' + sig.sentiment.fear_greed_index + ')';
    }
    if (sig.internals) {
      $('breadth').textContent = sig.internals.breadth_score.toFixed(2) + ' (TICK: ' + sig.internals.nyse_tick + ')';
    }
    if (sig.optionsai) {
      let oaiHtml = '';
      for (const [sym, oai] of Object.entries(sig.optionsai)) {
        const skewCls = oai.iv_skew > 0.005 ? 'green' : oai.iv_skew < -0.005 ? 'red' : '';
        const biasCls = oai.strategy_bias > 0.1 ? 'green' : oai.strategy_bias < -0.1 ? 'red' : '';
        oaiHtml += '<div style="margin-bottom:8px"><strong class="blue">' + sym + '</strong>';
        oaiHtml += '<div class="row"><span class="label">IV Skew</span><span class="val ' + skewCls + '">' + oai.iv_skew.toFixed(4) + '</span></div>';
        oaiHtml += '<div class="row"><span class="label">Expected Move</span><span class="val">$' + oai.move_amount.toFixed(2) + ' (' + oai.move_percent.toFixed(2) + '%)</span></div>';
        oaiHtml += '<div class="row"><span class="label">Implied Range</span><span class="val">' + oai.implied_low.toFixed(2) + ' - ' + oai.implied_high.toFixed(2) + '</span></div>';
        oaiHtml += '<div class="row"><span class="label">AI Bias</span><span class="val ' + biasCls + '">' + oai.strategy_bias.toFixed(2) + ' (' + oai.bullish_strategies + 'B/' + oai.bearish_strategies + 'S)</span></div>';
        if (oai.earnings_nearby) oaiHtml += '<div class="row"><span class="label">Earnings</span><span class="val red">NEARBY</span></div>';
        oaiHtml += '</div>';
      }
      $('optionsaiPanel').innerHTML = oaiHtml || '<span style="color:#8b949e">Waiting...</span>';
    }
  } catch(e) {}
}, 10000);

// Fetch activity once on load
(async () => {
  try {
    const r = await fetch('/api/activity');
    const trades = await r.json();
    if (!trades.length) { $('activity').innerHTML = '<span style="color:#8b949e">No trades yet</span>'; return; }
    $('activity').innerHTML = trades.slice(0, 25).map(t => {
      const pnl = Number(t.pnl || 0);
      const cls = pnl >= 0 ? 'green' : 'red';
      return '<div class="row"><span class="label">' + (t.exit_time || t.entry_time || '') + '</span><span>' + (t.underlying || '') + ' ' + (t.option_type || '') + ' $' + (t.strike || '') + '</span><span class="val ' + cls + '">' + (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) + '</span></div>';
    }).join('');
  } catch(e) { $('activity').innerHTML = '<span style="color:#8b949e">Could not load activity</span>'; }
})();
</script>
</body>
</html>"""
