import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a simple percentage grid and export a broker-style trade log CSV.")
    parser.add_argument("--data", required=True, help="Input CSV with minute bars")
    parser.add_argument("--entry-time", required=True, help="Initial buy timestamp, matching timestamp_exchange timezone")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Starting cash")
    parser.add_argument("--buy-fraction", type=float, default=0.5, help="Fraction of cash used for the first buy")
    parser.add_argument("--up-threshold", type=float, default=0.05, help="Sell when price rises this fraction from last fill")
    parser.add_argument("--down-threshold", type=float, default=0.05, help="Buy when price falls this fraction from last fill")
    parser.add_argument("--sell-position-fraction", type=float, default=0.5, help="Fraction of current position to sell")
    parser.add_argument("--buy-cash-fraction", type=float, default=1.0 / 3.0, help="Fraction of remaining cash to deploy on each buy")
    parser.add_argument("--contract", required=True, help="Contract label to write into the CSV, for example CLK26")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def load_bars(path: str) -> pd.DataFrame:
    bars = pd.read_csv(path)
    bars["ts"] = pd.to_datetime(bars["timestamp_exchange"])
    bars = bars.sort_values("ts").reset_index(drop=True)
    return bars


def style_timestamp(ts: pd.Timestamp) -> str:
    return ts.strftime("%m/%d/%Y %I:%M:%S %p")


def build_trade_row(
    ts: pd.Timestamp,
    contract: str,
    side: str,
    quantity: int,
    fill_price: float,
    realized_pnl: float,
    cash_after: float,
    position_after: int,
    avg_cost_after: float,
    reference_price_after: float,
) -> dict:
    return {
        "TRANSACTION DATE/TIME": style_timestamp(ts),
        "CONTRACT": contract,
        "PNL": round(realized_pnl, 2),
        "OPEN / CLOSE": "OPEN" if side == "BUY" else "CLOSE",
        "BUY / SELL": side,
        "QUANTITY": int(quantity),
        "ORDER TYPE": "MKT",
        "INSTRUMENT": "FUTURE",
        "CALL / PUT": "FALSE",
        "STRIKE PRICE": 0,
        "LIMIT PRICE": 0,
        "TIME IN FORCE": "DAY",
        "FILL PRICE": round(fill_price, 4),
        "STOP PRICE": 0,
        "CASH AFTER": round(cash_after, 2),
        "POSITION AFTER": int(position_after),
        "AVG COST AFTER": round(avg_cost_after, 4) if position_after else 0,
        "REFERENCE PRICE AFTER": round(reference_price_after, 4),
    }


def backtest(
    bars: pd.DataFrame,
    entry_time: str,
    initial_cash: float,
    buy_fraction: float,
    up_threshold: float,
    down_threshold: float,
    sell_position_fraction: float,
    buy_cash_fraction: float,
    contract: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    entry_ts = pd.Timestamp(entry_time)
    active = bars[bars["ts"] >= entry_ts].copy()
    if active.empty:
        raise ValueError("No bars found at or after the requested entry time")

    entry_rows = active[active["ts"] == entry_ts]
    if entry_rows.empty:
        raise ValueError("The requested entry time does not exist in the minute data")

    entry_bar = entry_rows.iloc[0]
    entry_idx = entry_rows.index[0]

    cash = float(initial_cash)
    position = 0
    avg_cost = 0.0
    reference_price = float(entry_bar["close"])
    trade_rows = []

    first_budget = cash * buy_fraction
    first_qty = int(first_budget // reference_price)
    if first_qty <= 0:
        raise ValueError("Initial cash is not enough to buy one unit at the entry price")
    first_cost = first_qty * reference_price
    cash -= first_cost
    position += first_qty
    avg_cost = reference_price
    trade_rows.append(
        build_trade_row(
            ts=entry_ts,
            contract=contract,
            side="BUY",
            quantity=first_qty,
            fill_price=reference_price,
            realized_pnl=0.0,
            cash_after=cash,
            position_after=position,
            avg_cost_after=avg_cost,
            reference_price_after=reference_price,
        )
    )

    for _, bar in active.loc[entry_idx + 1 :].iterrows():
        price = float(bar["close"])
        ts = bar["ts"]

        upper_trigger = reference_price * (1.0 + up_threshold)
        lower_trigger = reference_price * (1.0 - down_threshold)

        if price >= upper_trigger and position > 0:
            sell_qty = max(1, int(math.floor(position * sell_position_fraction)))
            realized_pnl = (price - avg_cost) * sell_qty
            cash += sell_qty * price
            position -= sell_qty
            if position == 0:
                avg_cost = 0.0
            reference_price = price
            trade_rows.append(
                build_trade_row(
                    ts=ts,
                    contract=contract,
                    side="SELL",
                    quantity=sell_qty,
                    fill_price=price,
                    realized_pnl=realized_pnl,
                    cash_after=cash,
                    position_after=position,
                    avg_cost_after=avg_cost,
                    reference_price_after=reference_price,
                )
            )
            continue

        if price <= lower_trigger and cash >= price:
            buy_budget = cash * buy_cash_fraction
            buy_qty = int(buy_budget // price)
            if buy_qty <= 0:
                continue
            cost = buy_qty * price
            new_position = position + buy_qty
            avg_cost = ((avg_cost * position) + (price * buy_qty)) / new_position
            cash -= cost
            position = new_position
            reference_price = price
            trade_rows.append(
                build_trade_row(
                    ts=ts,
                    contract=contract,
                    side="BUY",
                    quantity=buy_qty,
                    fill_price=price,
                    realized_pnl=0.0,
                    cash_after=cash,
                    position_after=position,
                    avg_cost_after=avg_cost,
                    reference_price_after=reference_price,
                )
            )

    last_bar = active.iloc[-1]
    summary = {
        "entry_time": style_timestamp(entry_ts),
        "entry_price": round(float(entry_bar["close"]), 4),
        "ending_time": style_timestamp(last_bar["ts"]),
        "ending_price": round(float(last_bar["close"]), 4),
        "ending_cash": round(cash, 2),
        "ending_position": int(position),
        "ending_avg_cost": round(avg_cost, 4) if position else 0,
        "ending_equity": round(cash + position * float(last_bar["close"]), 2),
        "trade_count": len(trade_rows),
    }
    return pd.DataFrame(trade_rows), summary


def main() -> int:
    args = parse_args()
    bars = load_bars(args.data)
    trade_log, summary = backtest(
        bars=bars,
        entry_time=args.entry_time,
        initial_cash=args.initial_cash,
        buy_fraction=args.buy_fraction,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        sell_position_fraction=args.sell_position_fraction,
        buy_cash_fraction=args.buy_cash_fraction,
        contract=args.contract,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trade_log.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"saved_to={output_path.resolve()}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
