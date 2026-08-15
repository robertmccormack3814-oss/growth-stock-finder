from __future__ import annotations

import io
import re
import pandas as pd
import requests
from common import DATA, load_json, save_json, now_iso

HEADERS = {"User-Agent": "Mozilla/5.0 GrowthStockFinder/1.0"}
ASX_ISIN_URL = "https://www.asx.com.au/content/dam/asx/issuers/ISIN.xls"
FALLBACK_URL = "https://stockanalysis.com/list/australian-securities-exchange/"

EXCLUDE = (
    "ETF", "EXCHANGE TRADED", "MANAGED FUND", "INDEX FUND", "BULLION",
    "WARRANT", "OPTION", "RIGHT", "PREFERENCE", "BOND", "NOTE"
)


def norm(c):
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def find_col(df, words):
    for c in df.columns:
        lc = norm(c)
        if any(w in lc for w in words):
            return c
    return None


def clean(v):
    return re.sub(r"[^A-Z0-9]", "", str(v or "").upper())


def official():
    r = requests.get(ASX_ISIN_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    raw = r.content
    frames = []
    try:
        frames.append(pd.read_excel(io.BytesIO(raw), engine="xlrd"))
    except Exception:
        pass
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            txt = raw.decode(enc)
        except Exception:
            continue
        for sep in ("\t", ",", "|"):
            try:
                df = pd.read_csv(io.StringIO(txt), sep=sep)
                if len(df) > 100 and len(df.columns) >= 2:
                    frames.append(df)
            except Exception:
                pass
    if not frames:
        raise RuntimeError("Could not parse official ASX company file")
    df = max(frames, key=len)
    code_col = find_col(df, ["asx code", "security code", "code"])
    name_col = find_col(df, ["company name", "issuer name", "issuer", "name"])
    desc_col = find_col(df, ["security description", "description", "security type"])
    if code_col is None or name_col is None:
        raise RuntimeError(f"Required ASX columns not found: {list(df.columns)}")
    out = {}
    for _, row in df.iterrows():
        code = clean(row.get(code_col))
        name = str(row.get(name_col) or "").strip()
        desc = str(row.get(desc_col) or "").strip() if desc_col else ""
        text = f" {name.upper()} {desc.upper()} "
        if not re.fullmatch(r"[A-Z0-9]{3}", code) or not name or name.lower() == "nan":
            continue
        if any(x in text for x in EXCLUDE):
            continue
        out[code] = {"symbol": code, "ticker": code + ".AX", "company": name, "source": "ASX", "updated_at": now_iso()}
    if len(out) < 500:
        raise RuntimeError(f"Official ASX list unexpectedly small: {len(out)}")
    return sorted(out.values(), key=lambda x: x["symbol"])


def fallback():
    tables = pd.read_html(FALLBACK_URL)
    df = max(tables, key=len)
    c = find_col(df, ["symbol"])
    n = find_col(df, ["company", "name"])
    if c is None or n is None:
        raise RuntimeError("Fallback list could not be parsed")
    out = {}
    for _, row in df.iterrows():
        code = clean(row[c])
        if re.fullmatch(r"[A-Z0-9]{3}", code):
            out[code] = {"symbol": code, "ticker": code + ".AX", "company": str(row[n]).strip(), "source": "fallback", "updated_at": now_iso()}
    return sorted(out.values(), key=lambda x: x["symbol"])


def main():
    old = load_json(DATA / "universe.json", [])
    try:
        universe = official()
        src = "official ASX"
    except Exception as exc:
        print("Official ASX universe failed:", exc)
        universe = fallback()
        src = "fallback"
    save_json(DATA / "universe.json", universe)
    if {x.get("symbol") for x in old} != {x.get("symbol") for x in universe}:
        save_json(DATA / "cursor.json", {"next_index": 0, "reset_at": now_iso()})
    print(f"Saved {len(universe)} ASX ordinary-equity candidates from {src}")


if __name__ == "__main__":
    main()
