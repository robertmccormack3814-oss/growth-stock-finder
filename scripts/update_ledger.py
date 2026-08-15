from __future__ import annotations

from datetime import datetime, timezone

from common import DATA, ROOT, load_json, now_iso, save_json

CFG = load_json(ROOT / "config.json", {})
POSITION_SIZE = float(CFG.get("ledger_position_size_usd", 1000))
MAX_RESULT_AGE_MINUTES = 120


def parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def fresh_result(row, now):
    dt = parse_iso(row.get("updated_at"))
    if not dt:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt.astimezone(timezone.utc)).total_seconds() <= MAX_RESULT_AGE_MINUTES * 60


def round2(v):
    return round(float(v), 2)


def mark_trade(trade, price):
    if price is None or price <= 0:
        return
    trade["current_price"] = round(float(price), 4)
    trade["current_value"] = round2(trade["units"] * float(price))
    trade["pnl"] = round2(trade["current_value"] - trade["initial_value"])
    trade["return_pct"] = round2((trade["current_value"] / trade["initial_value"] - 1) * 100)


def main():
    results = load_json(DATA / "results.json", {})
    ledger = load_json(DATA / "ledger.json", {"trades": [], "market": "NASDAQ", "currency": "USD", "created_at": now_iso()})
    trades = ledger.setdefault("trades", [])
    now = datetime.now(timezone.utc)

    open_by_symbol = {t["symbol"]: t for t in trades if t.get("status") == "OPEN"}

    for symbol, row in results.items():
        if row.get("market") not in (None, "NASDAQ") or not fresh_result(row, now):
            continue
        price = row.get("price")
        if price is None or price <= 0:
            continue

        trade = open_by_symbol.get(symbol)
        if trade:
            mark_trade(trade, price)
            trade["last_marked_at"] = row.get("updated_at") or now_iso()
            trade["latest_score"] = row.get("score")
            trade["latest_risk_score"] = row.get("risk_score")
            trade["latest_signal"] = row.get("status")

            if row.get("status") != "PASS":
                trade["status"] = "CLOSED"
                trade["exit_date"] = row.get("updated_at") or now_iso()
                trade["exit_price"] = round(float(price), 4)
                trade["exit_score"] = row.get("score")
                trade["exit_signal"] = row.get("status")
                entry_dt = parse_iso(trade.get("entry_date"))
                exit_dt = parse_iso(trade.get("exit_date"))
                if entry_dt and exit_dt:
                    trade["days_held"] = max(0, (exit_dt - entry_dt).days)
                open_by_symbol.pop(symbol, None)

        elif row.get("status") == "PASS":
            units = POSITION_SIZE / float(price)
            trade = {
                "id": f"{symbol}-{len(trades)+1}",
                "symbol": symbol,
                "company": row.get("company"),
                "market": "NASDAQ",
                "currency": "USD",
                "status": "OPEN",
                "entry_date": row.get("updated_at") or now_iso(),
                "entry_price": round(float(price), 4),
                "units": round(units, 8),
                "initial_value": round2(POSITION_SIZE),
                "entry_score": row.get("score"),
                "entry_risk_score": row.get("risk_score"),
                "entry_risk_grade": row.get("risk_grade"),
                "current_price": round(float(price), 4),
                "current_value": round2(POSITION_SIZE),
                "pnl": 0.0,
                "return_pct": 0.0,
                "latest_score": row.get("score"),
                "latest_risk_score": row.get("risk_score"),
                "latest_signal": "PASS",
                "last_marked_at": row.get("updated_at") or now_iso(),
            }
            trades.append(trade)
            open_by_symbol[symbol] = trade

    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
    all_pnl = sum(float(t.get("pnl") or 0) for t in trades)
    invested = sum(float(t.get("initial_value") or 0) for t in trades)
    wins = [t for t in closed_trades if float(t.get("pnl") or 0) > 0]
    losses = [t for t in closed_trades if float(t.get("pnl") or 0) < 0]
    returns = [float(t.get("return_pct") or 0) for t in closed_trades]

    ledger["market"] = "NASDAQ"
    ledger["currency"] = "USD"
    ledger["position_size_usd"] = POSITION_SIZE
    ledger["updated_at"] = now_iso()
    ledger["methodology"] = "Open a simulated US$1,000 equal-dollar position when a Nasdaq stock first becomes PASS; close it on the next fresh scan where it is no longer PASS. No brokerage, slippage, tax or dividends are included."
    ledger["summary"] = {
        "total_signals": len(trades),
        "open_positions": len(open_trades),
        "closed_trades": len(closed_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round2(len(wins) / len(closed_trades) * 100) if closed_trades else None,
        "average_closed_return_pct": round2(sum(returns) / len(returns)) if returns else None,
        "best_closed_return_pct": round2(max(returns)) if returns else None,
        "worst_closed_return_pct": round2(min(returns)) if returns else None,
        "realised_pnl": round2(sum(float(t.get("pnl") or 0) for t in closed_trades)),
        "unrealised_pnl": round2(sum(float(t.get("pnl") or 0) for t in open_trades)),
        "total_pnl": round2(all_pnl),
        "capital_allocated_across_signals": round2(invested),
        "aggregate_return_on_signal_capital_pct": round2(all_pnl / invested * 100) if invested else 0.0,
    }
    save_json(DATA / "ledger.json", ledger)
    print(f"NASDAQ ledger updated: {len(open_trades)} open, {len(closed_trades)} closed, P&L US${all_pnl:.2f}")


if __name__ == "__main__":
    main()
