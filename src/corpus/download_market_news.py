"""Downloader for real equity returns (yfinance) and financial headlines (Google News RSS)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

TICKERS = ("NVDA", "AMD", "INTC", "TSM", "ASML", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "QCOM")


def download_market_returns(output_csv: Path, period: str = "2y") -> pd.DataFrame:
    """Download daily close prices via yfinance and persist log/percentage returns CSV."""
    if output_csv.exists():
        return pd.read_csv(output_csv)

    raw = yf.download(list(TICKERS), period=period, auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    returns = closes.pct_change().dropna().reset_index()
    returns["Date"] = pd.to_datetime(returns["Date"]).dt.strftime("%Y-%m-%d")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_csv, index=False)
    return returns


def download_financial_news(output_json: Path) -> list[dict[str, str]]:
    """Fetch real financial headlines per ticker from Google News RSS and persist as JSON."""
    if output_json.exists():
        return json.loads(output_json.read_text(encoding="utf-8"))

    records: list[dict[str, str]] = []
    for ticker in TICKERS:
        records.extend(_fetch_ticker_rss(ticker))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _fetch_ticker_rss(ticker: str) -> list[dict[str, str]]:
    """Fetch up to 20 recent financial headlines for a single ticker."""
    url = f"https://news.google.com/rss/search?q={ticker}+stock+earnings&hl=en-US&gl=US&ceid=US:en"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    items: list[dict[str, str]] = []
    for item in root.findall(".//item")[:20]:
        title = item.findtext("title") or ""
        pub_date = item.findtext("pubDate") or ""
        items.append({"entity_id": ticker, "pub_date": pub_date, "headline": title})
    return items
