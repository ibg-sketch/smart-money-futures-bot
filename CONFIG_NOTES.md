# CONFIG_NOTES

This cheat sheet explains the most important configuration knobs from `config.yaml` without diving into the code. Use it to understand how the bot is wired and what to adjust for different trading attitudes.

## High-level structure
- **feature_flags** – enables data services and optional modules (data feeds, basis/UA funding snapshots, UIF features, AI analyst, order flow add-ons).
- **data_feeds / uif_engine** – runtime settings for background collectors (intervals, sinks, providers).
- **signals** – global signal behavior (minimum base confidence, SELL confirmation, dual-mode logic).
- **vwap_dev_sigma** – VWAP deviation thresholds (block/boost rules).
- **ttl** – adaptive TTL toggle.
- **safety** – kill switch.
- **report** – daily report schedule and optional quality gates.
- **default_coin / coin_configs** – per-asset weights, thresholds, targets, and ATR multipliers.
- **symbols** – list of actively traded tickers.

## What the main parameters mean
- **weights** (per coin): importance of each component in scoring (CVD, OI, VWAP, volume, liquidations, funding, RSI, EMA, basis_pct). UIF and order-flow weights are present but often zero/diagnostic unless explicitly enabled.
- **min_score_pct / min_score_pct_buy / min_score_pct_sell**: minimum share of the maximum score required to allow BUY/SELL for that asset. Lower = more permissive.
- **targets**: two numbers representing the min/max target move (%) used to derive price targets.
- **cvd_threshold**: absolute CVD magnitude required before CVD counts as directional support.
- **atr_multiplier**: scales ATR-based targets per symbol.
- **dev_sigma_thresholds**: VWAP deviation filters (block_below = ignore signals near VWAP; boost_above = allow extra boost for strong deviations).
- **volume_spike_mult** (global): factor for considering a volume spike vs median volume.
- **lookback_minutes / vwap_window**: historical depth for indicator calculations.
- **min_confidence**: baseline confidence guard-rail used when blending scores.
- **feature_flags.enable_order_flow / enable_uif_in_scoring**: toggle optional order-flow/UIF telemetry in scoring/formatting (weights may be zero by default).
- **report.time_utc**: UTC time for scheduled daily reports.

## Interpreting indicator-related settings
- **Volume spike / volume strength** – controlled by `volume_spike_mult` and per-coin volume weights. Higher spike multiplier means fewer spikes counted.
- **Open Interest (OI)** – per-coin weight; minimum % change thresholds are built into the logic. If `oi` weight is high, OI swings influence decisions more.
- **CVD** – `cvd_threshold` gate; values below the threshold are treated as noise. Formatting now shows `CVD: N/A` when data is stale/insignificant.
- **Basis / UIF** – `basis_pct` weight (default 0.10) influences UIF-30 integration; UIF weights default to 0 (telemetry only) unless you raise them.
- **VWAP / EMA / RSI** – per-coin weights plus global `vwap_dev_sigma` rules that block trades near VWAP and allow boosts far from it.
- **Regime handling** – regimes are detected automatically in the code; thresholds such as `vwap_dev_sigma.block_below` ensure consolidation zones are skipped.

## Tuning guidelines
- **More aggressive (more trades):**
  - Lower `min_score_pct_buy` / `min_score_pct_sell` for the specific coin.
  - Reduce `vwap_dev_sigma.block_below` slightly to allow signals closer to VWAP.
  - Lower `volume_spike_mult` to treat smaller volume upticks as spikes.
  - Increase weights for CVD/OI/volume a bit (small increments) to accept weaker confluence.
  - Shorten `lookback_minutes` if you want the model to react faster.

- **More conservative (fewer but higher-quality trades):**
  - Raise `min_score_pct_buy` / `min_score_pct_sell` for the coin.
  - Increase `vwap_dev_sigma.block_below` so consolidation is filtered out more often.
  - Increase `volume_spike_mult` to require stronger volume confirmation.
  - Decrease CVD/OI weights or raise `cvd_threshold` to demand clearer flow signals.
  - Keep `adaptive ttl` on and avoid lowering `min_confidence`.

## Where to look per coin
- Every symbol under `coin_configs` inherits from `default_coin` but overrides:
  - `weights`
  - `targets`
  - `cvd_threshold`
  - `atr_multiplier`
  - `dev_sigma_thresholds`
  - asymmetrical `min_score_pct_buy` / `min_score_pct_sell`

Adjust only the necessary fields for the target coin; unset fields fall back to `default_coin` values.
