"""
forecast_driver.py — Reproduce the deployed forward-fan forecast offline (no Firestore).

Run from the `prediction/` folder:   python forecast_driver.py

It (1) builds the inference frame from the same CSVs the training pipeline uses,
(2) VALIDATES by reproducing the Tabel 25 (CSA winners) test metrics, and
(3) prints the forward fan (t+1..t+7) for the example origin 2026-02-26 —
the numbers to paste into the "Contoh Output Prediksi Nyata" paragraph.

Trust the fan only if the printed validation MAPE/RMSE match Tabel 25.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))      # prediction/
ROOT = os.path.dirname(HERE)                            # project root
sys.path.insert(0, HERE)
os.chdir(HERE)

from feature_engineering import (                       # noqa: E402
    add_price_features, add_hmm_derived_features,
    engineer_all_features, select_feature_matrix,
)
import inference as INF                                 # noqa: E402

ANCHOR = pd.Timestamp("2026-02-26")   # example origin (last day of eval window)


def build_frame() -> pd.DataFrame:
    """Replicate inference.build_inference_frame() but source from CSVs."""
    price = pd.read_csv(f"{ROOT}/cpo/output/cpo_variables_Daily.csv", parse_dates=["Date"])
    price = price[["Date", "Close", "Open", "High", "Low", "Volume"]]
    sent = pd.read_csv(f"{ROOT}/news/output/sentiment_aggregate_Daily_title.csv", parse_dates=["Date"])
    sent = sent.rename(columns={
        "Combined_Positive_Prob": "Positive_Prob",
        "Combined_Negative_Prob": "Negative_Prob",
        "Combined_Neutral_Prob":  "Neutral_Prob",
    })[["Date", "Article_Count", "Positive_Prob", "Negative_Prob", "Neutral_Prob", "Sentiment_Score"]]
    hmm = pd.read_csv(f"{ROOT}/markov/output/hmm_states_results_Daily.csv", parse_dates=["Date"])
    hmm = hmm.rename(columns={"State": "HMM_State", "State_Label": "HMM_State_Label"})[
        ["Date", "HMM_State", "HMM_State_Label"]]

    # Clip so the last actual anchor is ANCHOR (2026-02-26).
    price = price[price["Date"] <= ANCHOR].sort_values("Date").reset_index(drop=True)
    sent = sent[sent["Date"] <= ANCHOR]
    hmm = hmm[hmm["Date"] <= ANCHOR]

    df = add_price_features(price)
    df = add_hmm_derived_features(df)
    df["Change_Pct"] = df["Close"].pct_change() * 100
    df = df.merge(hmm, on="Date", how="left")
    df["HMM_State"] = df["HMM_State"].fillna(2).astype(int)
    df["HMM_State_Label"] = df["HMM_State_Label"].fillna("Neutral")
    for state in df["HMM_State_Label"].value_counts().head(5).index.tolist():
        df[f"HMM_{state.replace(' ', '_').replace('-', '_')}"] = (df["HMM_State_Label"] == state).astype(int)
    df = df.drop(columns=["HMM_State_Label"])
    df = df.merge(sent, on="Date", how="left")
    for c in ["Article_Count", "Positive_Prob", "Negative_Prob", "Neutral_Prob", "Sentiment_Score"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    df["Confidence"] = df[["Positive_Prob", "Negative_Prob", "Neutral_Prob"]].max(axis=1).fillna(0.0)
    return engineer_all_features(df).sort_values("Date").reset_index(drop=True)


def predict(df: pd.DataFrame, i: int, h: int, winners: dict):
    tag = winners[h]
    model, scaler, fcols = INF.load_model(tag, h, variant="csa")
    row = df.iloc[[i]]
    try:
        X = select_feature_matrix(row, fcols)
    except Exception:
        return None
    if np.isnan(X).any():
        return None
    if scaler is not None:
        X = scaler.transform(X)
    lr = float(model.predict(X)[0])
    anchor_close = float(row["Close"].iloc[0])
    return anchor_close, anchor_close * np.exp(np.clip(lr, -10, 10)), lr


def main() -> None:
    df = build_frame()
    dates = df["Date"].tolist()
    last_idx = len(dates) - 1
    win = json.load(open("winners.json"))
    configs = {int(h): c for h, c in win["configs_by_horizon"].items()}
    winners = {int(h): t for h, t in win["winners_by_horizon"].items()}

    print("=== VALIDATION (compare to Tabel 25 CSA winners) ===")
    print(" h cfg   MAPE%     RMSE    nTest")
    test_start = pd.Timestamp("2025-01-01")
    for h in range(1, 8):
        preds, acts = [], []
        for i in range(len(df)):
            if dates[i] < test_start or i + h > last_idx:
                continue
            r = predict(df, i, h, winners)
            if r is None:
                continue
            preds.append(r[1]); acts.append(float(df["Close"].iloc[i + h]))
        preds, acts = np.array(preds), np.array(acts)
        mape = np.mean(np.abs((acts - preds) / acts)) * 100
        rmse = float(np.sqrt(np.mean((acts - preds) ** 2)))
        print(f" {h} {configs[h]:>3}  {mape:6.3f}  {rmse:8.2f}   {len(preds)}")

    print(f"\n=== FORWARD FAN — origin {ANCHOR.date()}  (anchor Close = {df['Close'].iloc[last_idx]:.0f} MYR/ton) ===")
    print(" h cfg   pred_logret   pred_price (MYR/ton)")
    for h in range(1, 8):
        r = predict(df, last_idx, h, winners)
        if r is None:
            print(f" {h} {configs[h]:>3}   <NaN feature>"); continue
        _, pp, lr = r
        print(f" {h} {configs[h]:>3}    {lr:+.5f}     {pp:8.1f}")


if __name__ == "__main__":
    main()
