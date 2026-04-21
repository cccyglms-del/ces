import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def fetch_chart(symbol: str, params: dict) -> dict:
    response = SESSION.get(YAHOO_CHART_URL.format(symbol=symbol), params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    error = payload.get("chart", {}).get("error")
    if error:
        raise RuntimeError(error.get("description") or error.get("code") or "Yahoo Finance returned an error")
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise RuntimeError("Yahoo Finance returned an empty chart result")
    return result[0]


def get_exchange_timezone(symbol: str) -> str:
    result = fetch_chart(symbol, {"interval": "1d", "range": "5d", "includePrePost": "true"})
    return result["meta"]["exchangeTimezoneName"]


def chart_to_frame(symbol: str, result: dict) -> pd.DataFrame:
    timestamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps:
        return pd.DataFrame(columns=["timestamp_utc", "timestamp_exchange", "open", "high", "low", "close", "volume", "symbol"])

    frame = pd.DataFrame(
        {
            "timestamp_epoch": timestamps,
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", []),
        }
    )
    exchange_tz = result["meta"]["exchangeTimezoneName"]
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_epoch"], unit="s", utc=True)
    frame["timestamp_exchange"] = frame["timestamp_utc"].dt.tz_convert(exchange_tz)
    frame["symbol"] = symbol
    frame = frame.dropna(subset=["open", "high", "low", "close"], how="all").reset_index(drop=True)
    return frame[["timestamp_utc", "timestamp_exchange", "open", "high", "low", "close", "volume", "symbol"]]


def fetch_intraday_range(symbol: str, start_date: str, end_date: str, interval: str, chunk_days: int) -> pd.DataFrame:
    exchange_tz = get_exchange_timezone(symbol)
    start_ts = pd.Timestamp(start_date, tz=exchange_tz)
    end_ts = pd.Timestamp(end_date, tz=exchange_tz)
    if end_ts <= start_ts:
        raise ValueError("end_date must be after start_date")

    frames = []
    cursor = start_ts
    delta = pd.Timedelta(days=chunk_days)
    while cursor < end_ts:
        chunk_end = min(cursor + delta, end_ts)
        params = {
            "interval": interval,
            "period1": int(cursor.timestamp()),
            "period2": int(chunk_end.timestamp()),
            "includePrePost": "true",
            "events": "div,splits",
        }
        result = fetch_chart(symbol, params)
        chunk = chart_to_frame(symbol, result)
        if not chunk.empty:
            frames.append(chunk)
        cursor = chunk_end
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "timestamp_exchange", "open", "high", "low", "close", "volume", "symbol"])

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    mask = (frame["timestamp_exchange"] >= start_ts) & (frame["timestamp_exchange"] < end_ts)
    frame = frame.loc[mask].reset_index(drop=True)
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    frame["timestamp_exchange"] = frame["timestamp_exchange"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance intraday futures data in chunked requests.")
    parser.add_argument("--symbol", required=True, help="Yahoo Finance symbol, for example MESM26.CME")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD, interpreted in the exchange timezone")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD, exclusive, interpreted in the exchange timezone")
    parser.add_argument("--interval", default="1m", help="Yahoo Finance interval, default: 1m")
    parser.add_argument("--chunk-days", type=int, default=7, help="Chunk size in days. Use 7 or 8 for 1m data.")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        frame = fetch_intraday_range(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            chunk_days=args.chunk_days,
        )
    except Exception as exc:
        print(f"Failed to fetch data: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    print(f"rows={len(frame)}")
    if not frame.empty:
        print(f"first_exchange={frame.iloc[0]['timestamp_exchange']}")
        print(f"last_exchange={frame.iloc[-1]['timestamp_exchange']}")
    print(f"saved_to={output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
