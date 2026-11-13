"""Reproduce the README calibration: synthetic 3x QQQ vs the real TQQQ fund.

Downloads QQQ, TQQQ and the 13-week T-bill rate (^IRX) from Yahoo Finance,
builds two synthetic series (the common simplified formula and sma200-bt's
borrow-cost formula), and compares final wealth over 2015-2024.

    pip install sma200-bt matplotlib
    python examples/calibrate_tqqq.py            # prints the table
    python examples/calibrate_tqqq.py --plot     # also writes docs/calibration_tqqq.png

Yahoo's adjusted prices get revised occasionally, so the last digit of the
multiples can drift between runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf

from sma200_bt import compound, fetch_tbill_rate, synthetic_letf_returns

START, END = "2015-01-01", "2025-01-01"  # END is exclusive in yfinance
TQQQ_ER = 0.0086
SPREAD = 0.0040


def adj_close(ticker: str):
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    s = df["Adj Close"]
    return s.iloc[:, 0] if getattr(s, "ndim", 1) == 2 else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="write docs/calibration_tqqq.png")
    args = ap.parse_args()

    qqq_ret = adj_close("QQQ").pct_change().dropna()
    tqqq_ret = adj_close("TQQQ").pct_change().dropna()
    idx = qqq_ret.index.intersection(tqqq_ret.index)
    qqq_ret, tqqq_ret = qqq_ret.loc[idx], tqqq_ret.loc[idx]

    tbill = fetch_tbill_rate(idx[0], idx[-1])
    simple = synthetic_letf_returns(qqq_ret, leverage=3.0, expense_ratio=TQQQ_ER, borrow_rate=0.0, borrow_spread=0.0)
    full = synthetic_letf_returns(qqq_ret, leverage=3.0, expense_ratio=TQQQ_ER, borrow_rate=tbill, borrow_spread=SPREAD)

    curves = {
        "Real TQQQ": compound(tqqq_ret),
        "Simple formula (no borrow cost)": compound(simple),
        f"sma200-bt (^IRX + {SPREAD * 1e4:.0f}bps)": compound(full),
    }
    real = curves["Real TQQQ"].iloc[-1]
    print(f"{idx[0].date()} to {idx[-1].date()}, $1 grows to:")
    for name, eq in curves.items():
        final = eq.iloc[-1]
        drift = "" if name == "Real TQQQ" else f"   drift vs real {final / real - 1:+.0%}"
        print(f"  {name:<36} {final:6.2f}x{drift}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
        styles = {
            "Real TQQQ": dict(color="#111111", lw=1.2, zorder=3),
            "Simple formula (no borrow cost)": dict(color="#c0392b", lw=1.4, ls="--"),
        }
        for name, eq in curves.items():
            ax.plot(eq.index, eq.values, label=f"{name}: {eq.iloc[-1]:.1f}x",
                    **styles.get(name, dict(color="#5dade2", lw=3.0, alpha=0.8)))
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $1 (log scale)")
        ax.set_title("Synthetic 3x QQQ vs real TQQQ, 2015-2024")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        out = Path(__file__).resolve().parent.parent / "docs" / "calibration_tqqq.png"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
