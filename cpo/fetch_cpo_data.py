"""
fetch_cpo_data.py
Fetches CPO (Palm Oil) futures historical data from Investing.com
using investiny and saves as Data_CPO_{Daily,Weekly,Monthly}.csv
matching the existing Indonesian-locale format.

Source: https://id.investing.com/commodities/palm-oil-historical-data
Instrument: Minyak Sawit Berjangka (FCPOc1) — Bursa Malaysia

Usage:
    pip install investiny
    python fetch_cpo_data.py

Optional: set CPO_INVESTING_ID below if auto-search is unreliable.
Run once with auto-search, note the printed ID, then hardcode it.
"""

import os
import pandas as pd
from datetime import datetime

CPO_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FROM  = "01/01/2015"          # m/d/Y — investiny format
DEFAULT_TO    = datetime.today().strftime("%m/%d/%Y")

# Investing.com numeric asset ID for "Minyak Sawit Berjangka (FCPOc1)"
# Set this after the first successful auto-search to avoid repeated lookups.
CPO_INVESTING_ID = None

INTERVALS = {
    "Daily":   "D"
    # "Weekly":  "W",
    # "Monthly": "M",
}


# ---------------------------------------------------------------------------
# Investing.com asset search
# ---------------------------------------------------------------------------

def find_cpo_id() -> int:
    """Search Investing.com for FCPOc1 (Minyak Sawit Berjangka) and return its numeric ID."""
    from investiny import search_assets

    candidates = []

    # Most specific queries first — FCPOc1 is the continuous front-month contract
    search_attempts = [
        dict(query="FCPOc1",         type="Commodities"),
        dict(query="CPOc1",   type="Commodities"),
        dict(query="FCPOc1"),
        dict(query="CPOc1"),
        # dict(query="Palm Oil",       type="Commodities"),
    ]

    for kwargs in search_attempts:
        try:
            results = search_assets(**kwargs)
            if results:
                candidates.extend(results)
                break
        except Exception:
            pass

    if not candidates:
        raise RuntimeError(
            "Could not find FCPOc1 on Investing.com via search.\n"
            "Set CPO_INVESTING_ID manually at the top of this script.\n"
            "You can find the ID in the page source of:\n"
            "  https://id.investing.com/commodities/palm-oil-historical-data"
        )

    print("Search results (showing top 5):")
    for i, r in enumerate(candidates[:5]):
        # Print raw dict so we can see the actual keys investiny returns
        print(f"  [{i}] {r}")

    # investiny may use 'id', 'investing_id', or a numeric key — find whichever exists
    def extract_id(r: dict):
        for key in ("id", "investing_id", "pairId", "pair_id"):
            if key in r and r[key] is not None:
                return int(r[key])
        # last resort: first integer value in the dict
        for v in r.values():
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        raise KeyError(f"Cannot find a numeric ID in result: {r}")

    # Prefer an exact symbol match if present
    fcpo_match = next(
        (r for r in candidates
         if any("FCPO" in str(r.get(k, "")).upper() for k in ("symbol", "name", "full_name"))),
        candidates[0],
    )
    asset_id = extract_id(fcpo_match)
    print(f"\nUsing: id={asset_id}  entry={fcpo_match}")
    print(f"Tip: hardcode CPO_INVESTING_ID = {asset_id} to skip this search next time.\n")
    return asset_id


# ---------------------------------------------------------------------------
# Number formatting helpers (Indonesian locale)
# ---------------------------------------------------------------------------

def _swap_separators(s: str) -> str:
    """Swap . and , so Python's default 1,234.56 → Indonesian 1.234,56"""
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_price(val) -> str:
    if pd.isna(val):
        return "-"
    return _swap_separators(f"{val:,.2f}")


def fmt_volume(val) -> str:
    """Return volume as Indonesian-formatted number with K suffix for ≥1000."""
    if pd.isna(val) or val == 0:
        return "-"
    if val >= 1_000:
        return _swap_separators(f"{val / 1_000:,.2f}") + "K"
    return _swap_separators(f"{val:,.2f}")


def fmt_change(val) -> str:
    if pd.isna(val):
        return "-"
    sign = "+" if val > 0 else ""
    return f"{sign}{_swap_separators(f'{val:.2f}')}%"


# ---------------------------------------------------------------------------
# Fetch & save one interval
# ---------------------------------------------------------------------------

def fetch_and_save(investing_id: int, interval_name: str, interval_code: int,
                   from_date: str, to_date: str, output_path: str) -> None:
    from investiny import historical_data

    print(f"\n  Fetching {interval_name}...", end=" ", flush=True)
    raw = historical_data(
        investing_id=investing_id,
        from_date=from_date,
        to_date=to_date,
        interval=interval_code,
    )

    df = pd.DataFrame(raw)
    # investiny returns columns: date, open, high, low, close, volume
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Change % relative to previous close
    df["change_pct"] = df["close"].pct_change() * 100

    # Output order: newest first (matches original CSVs)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "Tanggal":    row["date"].strftime("%d/%m/%Y"),
            "Terakhir":   fmt_price(row["close"]),
            "Pembukaan":  fmt_price(row["open"]),
            "Tertinggi":  fmt_price(row["high"]),
            "Terendah":   fmt_price(row["low"]),
            "Vol.":       fmt_volume(row.get("volume", float("nan"))),
            "Perubahan%": fmt_change(row["change_pct"]),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"{len(out_df)} rows saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(from_date: str = DEFAULT_FROM, to_date: str = DEFAULT_TO) -> None:
    global CPO_INVESTING_ID

    print(f"Date range: {from_date}  →  {to_date}\n")

    if CPO_INVESTING_ID is None:
        print("Searching Investing.com for CPO/Palm Oil Futures...")
        CPO_INVESTING_ID = find_cpo_id()

    for interval_name, interval_code in INTERVALS.items():
        output_path = os.path.join(CPO_DIR, f"Data_CPO_{interval_name}.csv")
        fetch_and_save(CPO_INVESTING_ID, interval_name, interval_code,
                       from_date, to_date, output_path)

    print("\nAll done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch CPO futures data from Investing.com")
    parser.add_argument("--from",  dest="from_date", default=DEFAULT_FROM,
                        help="Start date MM/DD/YYYY (default: %(default)s)")
    parser.add_argument("--to",    dest="to_date",   default=DEFAULT_TO,
                        help="End date MM/DD/YYYY (default: today)")
    parser.add_argument("--id",    dest="asset_id",  type=int, default=None,
                        help="Investing.com numeric asset ID (skip search)")
    args = parser.parse_args()

    if args.asset_id:
        CPO_INVESTING_ID = args.asset_id

    main(from_date=args.from_date, to_date=args.to_date)
