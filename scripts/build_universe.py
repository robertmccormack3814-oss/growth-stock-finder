from __future__ import annotations

import io
import re
import pandas as pd
import requests
from common import DATA, load_json, save_json, now_iso

HEADERS = {"User-Agent": "Mozilla/5.0 GrowthStockFinder/1.0"}
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

EXCLUDE_NAME_TERMS = (
    " ETF", "ETN", "WARRANT", "RIGHT", "UNIT", "PREFERRED", "PREFERENCE",
    "DEPOSITARY SHARE", "BENEFICIAL INTEREST", "ACQUISITION CORP", "SPAC"
)


def yahoo_symbol(symbol: str) -> str:
    # Yahoo generally represents class separators with a hyphen.
    return symbol.strip().upper().replace(".", "-")


def official_nasdaq():
    r = requests.get(NASDAQ_LISTED_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    df = pd.read_csv(io.StringIO(text), sep="|")

    required = {"Symbol", "Security Name", "Test Issue", "ETF"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Unexpected Nasdaq symbol-directory columns: {list(df.columns)}")

    out = {}
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol") or "").strip().upper()
        name = str(row.get("Security Name") or "").strip()
        if not symbol or symbol == "FILE CREATION TIME" or not name:
            continue
        if str(row.get("Test Issue") or "").strip().upper() == "Y":
            continue
        if str(row.get("ETF") or "").strip().upper() == "Y":
            continue
        if str(row.get("NextShares") or "").strip().upper() == "Y":
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", symbol):
            continue

        upper_name = " " + name.upper() + " "
        if any(term in upper_name for term in EXCLUDE_NAME_TERMS):
            continue

        out[symbol] = {
            "symbol": symbol,
            "ticker": yahoo_symbol(symbol),
            "company": name,
            "market_category": str(row.get("Market Category") or "").strip() or None,
            "financial_status": str(row.get("Financial Status") or "").strip() or None,
            "source": "Nasdaq Trader symbol directory",
            "market": "NASDAQ",
            "updated_at": now_iso(),
        }

    if len(out) < 1500:
        raise RuntimeError(f"Nasdaq universe unexpectedly small: {len(out)}")
    return sorted(out.values(), key=lambda x: x["symbol"])


def main():
    old = load_json(DATA / "universe.json", [])
    old_market = "ASX" if any(str(x.get("ticker", "")).endswith(".AX") for x in old[:50]) else None
    universe = official_nasdaq()
    save_json(DATA / "universe.json", universe)

    new_symbols = {x.get("symbol") for x in universe}
    old_symbols = {x.get("symbol") for x in old}
    market_changed = old_market == "ASX"

    if market_changed:
        # Prevent ASX signals and paper trades being mixed into the NASDAQ record.
        save_json(DATA / "results.json", {})
        save_json(DATA / "ranked.json", [])
        save_json(DATA / "state.json", {"signals": {}, "market": "NASDAQ", "reset_at": now_iso()})
        save_json(DATA / "ledger.json", {"trades": [], "market": "NASDAQ", "currency": "USD", "created_at": now_iso()})

    if market_changed or old_symbols != new_symbols:
        save_json(DATA / "cursor.json", {"next_index": 0, "reset_at": now_iso(), "market": "NASDAQ"})

    print(f"Saved {len(universe)} Nasdaq-listed equity candidates from Nasdaq Trader")
    if market_changed:
        print("Market changed from ASX to NASDAQ: old results and ledger were reset for a clean record")


if __name__ == "__main__":
    main()
