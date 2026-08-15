from __future__ import annotations

import json
import math
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from common import DATA, ROOT, clamp, load_json, now_iso, safe_float, save_json

CFG = load_json(ROOT / "config.json", {})
WEIGHTS = CFG.get("weights", {"growth": 30, "quality": 25, "balance_sheet": 15, "momentum": 20, "valuation": 10})


def pct(v):
    x = safe_float(v)
    return None if x is None else x * 100.0


def row_value(df: pd.DataFrame, names, col=0):
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index and len(df.columns) > col:
            return safe_float(df.loc[name].iloc[col])
    return None


def series_from_row(df: pd.DataFrame, names):
    if df is None or df.empty:
        return []
    for name in names:
        if name in df.index:
            vals = [safe_float(x) for x in df.loc[name].tolist()]
            return [x for x in vals if x is not None]
    return []


def cagr(values):
    vals = [v for v in values if v is not None and v > 0]
    if len(vals) < 3:
        return None
    newest, oldest = vals[0], vals[-1]
    years = len(vals) - 1
    if oldest <= 0 or years <= 0:
        return None
    try:
        return (newest / oldest) ** (1 / years) - 1
    except Exception:
        return None


def score_linear(v, bad, good, inverse=False):
    if v is None:
        return 0.35
    if inverse:
        return clamp((bad - v) / (bad - good)) if bad != good else 0
    return clamp((v - bad) / (good - bad)) if good != bad else 0


def annualised_vol(prices):
    r = prices.pct_change().dropna()
    if len(r) < 30:
        return None
    return float(r.std() * math.sqrt(252))


def max_drawdown(prices):
    if len(prices) < 30:
        return None
    peak = prices.cummax()
    dd = prices / peak - 1
    return float(dd.min())


def analyse(stock, benchmark_close):
    ticker = stock["ticker"]
    t = yf.Ticker(ticker)
    info = t.get_info() or {}
    quote_type = str(info.get("quoteType") or "").upper()
    if quote_type and quote_type not in ("EQUITY",):
        raise ValueError(f"quoteType={quote_type}")

    hist = t.history(period="2y", auto_adjust=True)
    if hist.empty or len(hist) < 210:
        raise ValueError("insufficient price history")
    close = hist["Close"].dropna()
    volume = hist["Volume"].fillna(0)
    price = float(close.iloc[-1])
    market_cap = safe_float(info.get("marketCap"))
    avg_dollar_vol = float((close.tail(60) * volume.tail(60)).mean())
    if price < CFG.get("min_price_aud", 0.2):
        raise ValueError("below minimum price")
    if market_cap is not None and market_cap < CFG.get("min_market_cap_aud", 100_000_000):
        raise ValueError("below minimum market cap")
    if avg_dollar_vol < CFG.get("min_avg_dollar_volume_aud", 250_000):
        raise ValueError("below minimum liquidity")

    inc = t.get_income_stmt(freq="yearly")
    bal = t.get_balance_sheet(freq="yearly")
    cf = t.get_cashflow(freq="yearly")

    revenues = series_from_row(inc, ["Total Revenue", "Operating Revenue"])
    diluted_eps = series_from_row(inc, ["Diluted EPS", "Basic EPS"])
    revenue_cagr = cagr(revenues[:4])
    eps_cagr = cagr([x for x in diluted_eps[:4] if x > 0])
    rev_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth")) or safe_float(info.get("earningsQuarterlyGrowth"))

    roe = safe_float(info.get("returnOnEquity"))
    roa = safe_float(info.get("returnOnAssets"))
    gross_margin = safe_float(info.get("grossMargins"))
    op_margin = safe_float(info.get("operatingMargins"))
    profit_margin = safe_float(info.get("profitMargins"))
    fcf = safe_float(info.get("freeCashflow"))
    revenue_ttm = safe_float(info.get("totalRevenue"))
    fcf_margin = (fcf / revenue_ttm) if fcf is not None and revenue_ttm not in (None, 0) else None

    ebit = row_value(inc, ["EBIT", "Operating Income"])
    tax = row_value(inc, ["Tax Provision"])
    pretax = row_value(inc, ["Pretax Income"])
    tax_rate = clamp(tax / pretax, 0, 0.35) if tax is not None and pretax and pretax > 0 else 0.30
    equity = row_value(bal, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    debt = row_value(bal, ["Total Debt"])
    cash = row_value(bal, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    invested_capital = None
    if equity is not None:
        invested_capital = equity + (debt or 0) - (cash or 0)
    roic = (ebit * (1 - tax_rate) / invested_capital) if ebit is not None and invested_capital and invested_capital > 0 else None

    current_ratio = safe_float(info.get("currentRatio"))
    debt_to_equity = safe_float(info.get("debtToEquity"))
    if debt_to_equity is not None:
        debt_to_equity /= 100.0
    ebitda = safe_float(info.get("ebitda"))
    net_debt_ebitda = ((debt or 0) - (cash or 0)) / ebitda if ebitda and ebitda > 0 else None
    interest = row_value(inc, ["Interest Expense", "Interest Expense Non Operating"])
    interest_cover = abs(ebit / interest) if ebit is not None and interest not in (None, 0) else None

    ma50 = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean())
    ret6 = price / float(close.iloc[-126]) - 1 if len(close) >= 126 else None
    ret12 = price / float(close.iloc[-252]) - 1 if len(close) >= 252 else None
    high52 = float(close.tail(252).max())
    near_high = price / high52
    vol_ratio = float(volume.tail(20).mean() / max(volume.tail(100).mean(), 1))

    benchmark = benchmark_close.reindex(close.index).ffill().dropna()
    rs12 = None
    if len(benchmark) >= 252 and len(close) >= 252:
        bret = float(benchmark.iloc[-1] / benchmark.iloc[-252] - 1)
        rs12 = ret12 - bret

    forward_pe = safe_float(info.get("forwardPE"))
    trailing_pe = safe_float(info.get("trailingPE"))
    peg = safe_float(info.get("trailingPegRatio")) or safe_float(info.get("pegRatio"))
    fcf_yield = fcf / market_cap if fcf is not None and market_cap and market_cap > 0 else None

    growth_parts = [
        score_linear(rev_growth, 0.05, 0.30),
        score_linear(earnings_growth, 0.05, 0.35),
        score_linear(revenue_cagr, 0.05, 0.25),
        score_linear(eps_cagr, 0.05, 0.30),
    ]
    growth_score = float(np.mean(growth_parts)) * WEIGHTS["growth"]

    quality_parts = [
        score_linear(roe, 0.08, 0.25),
        score_linear(roic, 0.06, 0.20),
        score_linear(roa, 0.03, 0.12),
        score_linear(gross_margin, 0.20, 0.60),
        score_linear(op_margin, 0.05, 0.25),
        score_linear(fcf_margin, 0.02, 0.18),
    ]
    quality_score = float(np.mean(quality_parts)) * WEIGHTS["quality"]

    balance_parts = [
        score_linear(current_ratio, 0.8, 2.0),
        score_linear(debt_to_equity, 1.5, 0.25, inverse=True),
        score_linear(net_debt_ebitda, 3.5, 0.5, inverse=True),
        score_linear(interest_cover, 2.0, 10.0),
    ]
    balance_score = float(np.mean(balance_parts)) * WEIGHTS["balance_sheet"]

    momentum_parts = [
        1.0 if price > ma50 else 0.0,
        1.0 if price > ma200 else 0.0,
        score_linear(ret6, -0.05, 0.35),
        score_linear(rs12, -0.10, 0.25),
        score_linear(near_high, 0.70, 0.98),
        score_linear(vol_ratio, 0.7, 1.5),
    ]
    momentum_score = float(np.mean(momentum_parts)) * WEIGHTS["momentum"]

    growth_ref = max([x for x in (earnings_growth, revenue_cagr, rev_growth) if x is not None] or [0.10])
    pe_to_growth = (forward_pe or trailing_pe) / max(growth_ref * 100, 1) if (forward_pe or trailing_pe) else None
    valuation_parts = [
        score_linear(peg, 3.0, 1.0, inverse=True),
        score_linear(pe_to_growth, 2.5, 0.8, inverse=True),
        score_linear(fcf_yield, 0.00, 0.06),
    ]
    valuation_score = float(np.mean(valuation_parts)) * WEIGHTS["valuation"]

    total_score = round(growth_score + quality_score + balance_score + momentum_score + valuation_score, 1)

    vol = annualised_vol(close)
    mdd = max_drawdown(close)
    beta = safe_float(info.get("beta"))
    risk_parts = [
        score_linear(vol, 0.20, 0.65),
        score_linear(abs(mdd) if mdd is not None else None, 0.15, 0.55),
        score_linear(beta, 0.8, 1.8),
        score_linear(debt_to_equity, 0.25, 1.5),
        score_linear(pe_to_growth, 0.8, 3.0),
        score_linear(100_000_000 / market_cap if market_cap else None, 0.0, 1.0),
    ]
    risk_score = round(float(np.mean(risk_parts)) * 100, 1)
    if risk_score < 30:
        risk_grade = "Low"
    elif risk_score < 50:
        risk_grade = "Moderate"
    elif risk_score < 70:
        risk_grade = "High"
    else:
        risk_grade = "Very High"

    growth_gate = (rev_growth is not None and rev_growth >= 0.10) or (earnings_growth is not None and earnings_growth >= 0.15) or (revenue_cagr is not None and revenue_cagr >= 0.10)
    cash_gate = (fcf is not None and fcf > 0) or (profit_margin is not None and profit_margin > 0)
    trend_gate = price > ma200
    risk_gate = risk_score <= CFG.get("max_risk_score_for_pass", 65)
    passed = total_score >= CFG.get("pass_score", 78) and growth_gate and cash_gate and trend_gate and risk_gate
    watch = (not passed) and total_score >= CFG.get("watch_score", 65)
    status = "PASS" if passed else "WATCH" if watch else "NO SIGNAL"

    return {
        "symbol": stock["symbol"], "ticker": ticker, "company": info.get("longName") or stock.get("company"),
        "price": round(price, 3), "market_cap": market_cap, "avg_dollar_volume": round(avg_dollar_vol, 0),
        "score": total_score, "status": status, "risk_score": risk_score, "risk_grade": risk_grade,
        "component_scores": {"growth": round(growth_score, 1), "quality": round(quality_score, 1), "balance_sheet": round(balance_score, 1), "momentum": round(momentum_score, 1), "valuation": round(valuation_score, 1)},
        "metrics": {
            "revenue_growth": pct(rev_growth), "earnings_growth": pct(earnings_growth), "revenue_cagr": pct(revenue_cagr), "eps_cagr": pct(eps_cagr),
            "roe": pct(roe), "roic": pct(roic), "gross_margin": pct(gross_margin), "operating_margin": pct(op_margin), "fcf_margin": pct(fcf_margin),
            "current_ratio": current_ratio, "debt_to_equity": debt_to_equity, "net_debt_ebitda": net_debt_ebitda, "interest_cover": interest_cover,
            "return_6m": pct(ret6), "return_12m": pct(ret12), "relative_strength_12m": pct(rs12), "distance_to_52w_high": pct(near_high - 1),
            "forward_pe": forward_pe, "peg": peg, "fcf_yield": pct(fcf_yield), "volatility": pct(vol), "max_drawdown": pct(mdd), "beta": beta
        },
        "gates": {"growth": growth_gate, "cash_profit": cash_gate, "above_200dma": trend_gate, "risk": risk_gate},
        "updated_at": now_iso()
    }


def email_alert(rows):
    user = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_APP_PASSWORD")
    if not user or not password or not rows:
        return False
    body = ["Growth Stock Finder found new PASS signals:\n"]
    for r in rows:
        body.append(f"{r['symbol']} — {r['company']}\nScore {r['score']}/100 | Risk {r['risk_score']}/100 ({r['risk_grade']}) | Price ${r['price']}\nGrowth {r['component_scores']['growth']}/{WEIGHTS['growth']} | Quality {r['component_scores']['quality']}/{WEIGHTS['quality']} | Momentum {r['component_scores']['momentum']}/{WEIGHTS['momentum']}\n")
    body.append("\nA PASS is a screening signal, not a recommendation. Review company announcements, valuation and position sizing before acting.")
    msg = MIMEText("\n".join(body))
    msg["Subject"] = f"Growth Stock Finder: {len(rows)} new PASS signal{'s' if len(rows) != 1 else ''}"
    msg["From"] = user
    msg["To"] = user
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(user, password)
        server.send_message(msg)
    return True


def main():
    universe = load_json(DATA / "universe.json", [])
    if not universe:
        raise SystemExit("No universe.json. Run build_universe.py first.")
    old_results = load_json(DATA / "results.json", {})
    state = load_json(DATA / "state.json", {"signals": {}})
    cursor = load_json(DATA / "cursor.json", {"next_index": 0})
    start = int(cursor.get("next_index", 0)) % len(universe)
    size = int(os.getenv("BATCH_SIZE") or CFG.get("batch_size", 75))
    batch = [universe[(start + i) % len(universe)] for i in range(min(size, len(universe)))]

    bench = yf.download(CFG.get("benchmark", "^AXJO"), period="2y", auto_adjust=True, progress=False)
    if bench.empty:
        raise SystemExit("Could not download ASX benchmark")
    bclose = bench["Close"]
    if isinstance(bclose, pd.DataFrame):
        bclose = bclose.iloc[:, 0]

    new_alerts = []
    for stock in batch:
        symbol = stock["symbol"]
        try:
            result = analyse(stock, bclose)
            old_results[symbol] = result
            prior = state.setdefault("signals", {}).get(symbol, {})
            was_pass = prior.get("status") == "PASS"
            if result["status"] == "PASS" and not was_pass:
                new_alerts.append(result)
            state["signals"][symbol] = {"status": result["status"], "score": result["score"], "risk_score": result["risk_score"], "updated_at": result["updated_at"]}
            print(symbol, result["status"], result["score"], "risk", result["risk_score"])
        except Exception as exc:
            old_results[symbol] = {"symbol": symbol, "company": stock.get("company"), "status": "ERROR", "error": str(exc), "updated_at": now_iso()}
            print(symbol, "ERROR", exc)

    ranked = sorted(old_results.values(), key=lambda r: (r.get("status") == "PASS", r.get("status") == "WATCH", r.get("score", -1)), reverse=True)
    save_json(DATA / "results.json", {r["symbol"]: r for r in ranked})
    save_json(DATA / "ranked.json", ranked)
    save_json(DATA / "state.json", state)
    save_json(DATA / "cursor.json", {"next_index": (start + len(batch)) % len(universe), "last_run": now_iso(), "processed": len(batch), "universe_size": len(universe)})
    if new_alerts:
        sent = email_alert(new_alerts)
        print(f"New PASS alerts: {len(new_alerts)}; email sent={sent}")
    else:
        print("No new PASS alerts")


if __name__ == "__main__":
    main()
