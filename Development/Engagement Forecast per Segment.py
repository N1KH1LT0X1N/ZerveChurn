import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Zerve design system (all private _ vars to avoid upstream conflicts)
_BG      = '#1D1D20'
_TXT     = '#fbfbff'
_SGRAY   = '#909094'
_FC_CLRS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#ffd400', '#17b26a']

print("=" * 70)
print("ENGAGEMENT TIME-SERIES FORECASTING PER USER SEGMENT")
print("Using: Double Exponential Smoothing (Holt's Linear Trend)")
print("=" * 70)

# ── Pure-numpy Holt double exponential smoothing
def _holt_forecast(y, h, alpha=None, beta=None):
    """
    Holt's linear method.
    If alpha/beta are provided they are used directly; otherwise a grid
    search finds the optimal pair.
    """
    n = len(y)

    # FIX 1: honour caller-supplied alpha / beta instead of always grid-searching
    if alpha is not None and beta is not None:
        best_a, best_b = alpha, beta
    else:
        best_sse, best_a, best_b = np.inf, 0.3, 0.1
        for _a in np.arange(0.1, 0.95, 0.1):
            for _b in np.arange(0.01, 0.5, 0.05):
                L, T = y[0], y[1] - y[0]
                sse = 0.0
                for t in range(1, n):
                    pred = L + T
                    sse += (y[t] - pred) ** 2
                    L_new = _a * y[t] + (1 - _a) * (L + T)
                    T     = _b * (L_new - L) + (1 - _b) * T
                    L     = L_new
                if sse < best_sse:
                    best_sse, best_a, best_b = sse, _a, _b

    # Refit with best params & collect residuals
    L, T = y[0], y[1] - y[0]
    resids = []
    for t in range(1, n):
        pred  = L + T
        resids.append(y[t] - pred)
        L_new = best_a * y[t] + (1 - best_a) * (L + T)
        T     = best_b * (L_new - L) + (1 - best_b) * T
        L     = L_new

    resid_std = float(np.std(resids)) if resids else float(np.std(y))
    fc = np.array([max(0.0, L + (k + 1) * T) for k in range(h)])
    return fc, resid_std, best_a, best_b


# ── Build segment-level daily engagement time series
_seg_map = master_segments[['distinct_id', 'engagement_segment']].set_index('distinct_id')['engagement_segment']

_daily_fc = daily_counts.copy()
_daily_fc['date']    = pd.to_datetime(_daily_fc['date'])
_daily_fc['segment'] = _daily_fc['distinct_id'].map(_seg_map).fillna('Unknown')
_daily_fc = _daily_fc[_daily_fc['segment'] != 'Unknown']

seg_daily_agg = (
    _daily_fc.groupby(['segment', 'date'])['daily_events']
    .sum()
    .reset_index()
    .sort_values(['segment', 'date'])
)

_segments_list = sorted(seg_daily_agg['segment'].unique())
print(f"\nSegments found: {_segments_list}")
print(f"Date range: {seg_daily_agg['date'].min().date()} → {seg_daily_agg['date'].max().date()}")
print(f"Total segment-days: {len(seg_daily_agg)}")

_global_min = seg_daily_agg['date'].min()
_global_max = seg_daily_agg['date'].max()
_full_idx   = pd.date_range(_global_min, _global_max, freq='D')
_n_history  = len(_full_idx)
print(f"History: {_n_history} days of data")

# ── Forecast horizons
_horizons       = [30, 60, 90]
_last_date      = _global_max
_forecast_dates = {h: pd.date_range(_last_date + pd.Timedelta(days=1), periods=h, freq='D')
                   for h in _horizons}

# ── Fit per segment
print("\n[FITTING MODELS — Holt Double Exponential Smoothing]")
_all_fc_records = []
_seg_models     = {}

for _si, _seg in enumerate(_segments_list):
    _ts_raw = seg_daily_agg[seg_daily_agg['segment'] == _seg].set_index('date')['daily_events']
    _ts     = _ts_raw.reindex(_full_idx, fill_value=0).astype(float)

    # FIX 2: center=False avoids partial-future windows distorting the right edge
    _ts_smo = _ts.rolling(7, min_periods=1, center=False).mean()
    _yvals  = _ts.values
    _n_obs  = len(_yvals)

    print(f"  {_seg}: {_n_obs} days | mean={_yvals.mean():.1f} | std={_yvals.std():.1f}")

    _fc_means = {}
    _fc_lower = {}
    _fc_upper = {}

    if _n_obs >= 4:
        for h in _horizons:
            _fc_vals, _rs, _a, _b = _holt_forecast(_yvals, h)
            # Forecast variance grows with horizon: σ²·(1 + h/n)
            _ci_half      = 1.96 * _rs * np.sqrt(1 + np.arange(1, h + 1) / _n_obs)
            _fc_means[h]  = _fc_vals
            _fc_lower[h]  = np.clip(_fc_vals - _ci_half, 0, None)
            _fc_upper[h]  = _fc_vals + _ci_half
    else:
        _flat = float(_yvals.mean()) if _yvals.sum() > 0 else 0.0
        # FIX 3: ensure a meaningful minimum std so CI bands are always visible
        _rs   = max(float(_yvals.std()), 1.0)
        for h in _horizons:
            _fc_means[h] = np.full(h, _flat)
            _fc_lower[h] = np.clip(np.full(h, _flat - 1.96 * _rs), 0, None)
            _fc_upper[h] = np.full(h, _flat + 1.96 * _rs)

    _seg_models[_seg] = {
        'ts': _ts, 'ts_smo': _ts_smo,
        'fc_means': _fc_means, 'fc_lower': _fc_lower, 'fc_upper': _fc_upper,
    }

    for h in _horizons:
        for _fi, _fd in enumerate(_forecast_dates[h]):
            _all_fc_records.append({
                'segment':              _seg,
                'horizon':              h,
                'forecast_date':        _fd,
                'predicted_engagement': round(float(_fc_means[h][_fi]), 4),
                'lower_bound':          round(float(_fc_lower[h][_fi]), 4),
                'upper_bound':          round(float(_fc_upper[h][_fi]), 4),
            })

# ── Output dataframe
engagement_forecast = (
    pd.DataFrame(_all_fc_records)
    .sort_values(['segment', 'horizon', 'forecast_date'])
    .reset_index(drop=True)
)

print(f"\n[OUTPUT] engagement_forecast: {engagement_forecast.shape}")
print(engagement_forecast.head(12).to_string(index=False))
print(f"\nSegments: {engagement_forecast['segment'].nunique()} | Horizons: {sorted(engagement_forecast['horizon'].unique())} days")

_fc_summary = (
    engagement_forecast
    .groupby(['segment', 'horizon'])['predicted_engagement']
    .agg(['mean', 'min', 'max'])
    .round(2)
)
print("\nMean predicted engagement per segment × horizon:")
print(_fc_summary.to_string())

# ── Per-segment forecast charts with confidence intervals
print("\n[VISUALIZING FORECASTS WITH CONFIDENCE INTERVALS]")
_h_alpha = {90: 0.12, 60: 0.18, 30: 0.25}
_h_lw    = {90: 1.2,  60: 1.5,  30: 2.0}
_h_ls    = {90: ':',  60: '--', 30: '-'}

for _pi, _pseg in enumerate(_segments_list):
    _psm  = _seg_models[_pseg]
    _pts  = _psm['ts']
    _pclr = _FC_CLRS[_pi % len(_FC_CLRS)]

    fig_seg = plt.figure(figsize=(14, 5), facecolor=_BG)
    _pax = fig_seg.add_subplot(1, 1, 1)
    _pax.set_facecolor(_BG)

    _pax.plot(_full_idx, _pts.values,          color=_pclr, alpha=0.2,  linewidth=1,   label='Daily (actual)')
    _pax.plot(_full_idx, _psm['ts_smo'].values, color=_pclr, alpha=0.85, linewidth=1.8, label='7-day smoothed')

    for _ph in sorted(_horizons, reverse=True):
        _pfd = _forecast_dates[_ph]

        # FIX 4 (revised): stitch the last historical date onto the forecast
        # horizon as a real DatetimeIndex. The previous np.concatenate path
        # produced an object-dtype array mixing pd.Timestamp + datetime64[ns],
        # which matplotlib then tried to interpret as numeric ordinals and
        # blew up with "Date ordinal -1004651.4 ..." inside plt.tight_layout()
        # (see docs/repo_state_and_next_steps.md Blocker C).
        _pstch = pd.DatetimeIndex([_full_idx[-1]]).append(pd.DatetimeIndex(_pfd))
        _pmean = np.concatenate([[float(_pts.iloc[-1])], _psm['fc_means'][_ph]])
        _plo   = np.concatenate([[float(_pts.iloc[-1])], _psm['fc_lower'][_ph]])
        # FIX 5: renamed _phi → _pup to avoid visual confusion with loop var _ph
        _pup   = np.concatenate([[float(_pts.iloc[-1])], _psm['fc_upper'][_ph]])

        _pax.fill_between(_pstch, _plo, _pup, color=_pclr, alpha=_h_alpha[_ph])
        _pax.plot(_pstch, _pmean, color=_pclr, linewidth=_h_lw[_ph],
                  linestyle=_h_ls[_ph], label=f'{_ph}d forecast')

    _pax.axvline(_global_max, color=_SGRAY, linestyle='--', linewidth=1, alpha=0.6)
    _pyl = _pax.get_ylim()
    _pax.text(_global_max, _pyl[1] * 0.97, '  Forecast →', color=_SGRAY, fontsize=9, va='top')
    _pax.set_title(f'Engagement Forecast — {_pseg}', color=_TXT, fontsize=14, fontweight='bold', pad=12)
    _pax.set_xlabel('Date',                              color=_TXT, fontsize=11)
    _pax.set_ylabel('Daily Events (segment total)',      color=_TXT, fontsize=11)
    _pax.tick_params(colors=_TXT, labelsize=9)
    for _spk in ['top', 'right']:
        _pax.spines[_spk].set_visible(False)
    _pax.spines['bottom'].set_color(_SGRAY)
    _pax.spines['left'].set_color(_SGRAY)
    _pax.legend(loc='upper left', fontsize=9, framealpha=0.3,
                facecolor=_BG, edgecolor=_SGRAY, labelcolor=_TXT)
    plt.tight_layout()
    print(f"  ✓ {_pseg}")
    plt.show()

# ── 90-day comparison summary bar chart
fig_fc_summary = plt.figure(figsize=(13, 5), facecolor=_BG)
_ax_fc = fig_fc_summary.add_subplot(1, 1, 1)
_ax_fc.set_facecolor(_BG)

_sum90 = (
    engagement_forecast[engagement_forecast['horizon'] == 90]
    .groupby('segment')['predicted_engagement'].mean()
    .sort_values(ascending=False)
)
_bx     = np.arange(len(_sum90))
_brects = _ax_fc.bar(
    _bx, _sum90.values,
    color=[_FC_CLRS[_bi % len(_FC_CLRS)] for _bi in range(len(_sum90))],
    width=0.55, edgecolor='none', alpha=0.85
)
for _br in _brects:
    _bh = _br.get_height()
    _ax_fc.text(_br.get_x() + _br.get_width() / 2, _bh + 0.5,
                f'{_bh:.0f}', ha='center', va='bottom', color=_TXT, fontsize=10)

_ax_fc.set_xticks(_bx)
_ax_fc.set_xticklabels(_sum90.index, color=_TXT, fontsize=10)
_ax_fc.set_ylabel('Mean Daily Events (90-day forecast)', color=_TXT, fontsize=11)
_ax_fc.set_title('90-Day Average Forecast Engagement by Segment',
                 color=_TXT, fontsize=14, fontweight='bold', pad=12)
_ax_fc.tick_params(colors=_TXT)
for _spk in ['top', 'right']:
    _ax_fc.spines[_spk].set_visible(False)
_ax_fc.spines['bottom'].set_color(_SGRAY)
_ax_fc.spines['left'].set_color(_SGRAY)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("✓ FORECASTING COMPLETE")
print(f"  engagement_forecast: {len(engagement_forecast)} rows × {len(engagement_forecast.columns)} cols")
print(f"  Segments: {list(engagement_forecast['segment'].unique())}")
print(f"  Horizons: {sorted(engagement_forecast['horizon'].unique())} days")
print("=" * 70)