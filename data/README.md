# Data Documentation

## Directory Overview

### `proxy_records/`
Raw paleoclimate proxy time series (CSV format, 2 columns: age [ka], proxy value).

**Naming**: file names preserve the original dataset identifiers (site/core and
proxy) verbatim — including spaces, Greek letters and degree signs — so each
file maps one-to-one onto the record names in the paper's Table S1/S2.

Examples:
- `2_ODP983-IRD.csv` — ODP 983 (North Atlantic) ice-rafted debris
- `32_Jouzel-2007-dD.csv` — EPICA Dome C δD (Jouzel et al., 2007)
- `16-1_ODP 184-1143-UK37_SST=800ka.csv` — ODP 1143 UK'37 SST, 0–800 ka
- `14_cheng2016composite-noaa.xlsx` — Cheng et al. (2016) speleothem composite (xlsx)

**Important**: Some files have headers, some don't. The `src/io.py` module auto-detects
the age and value columns.

**File count**: 83 data files (80 CSV + 3 XLSX) covering 33 numbered site groups.

### `proxy_errors/`
Age-model 1σ uncertainty files. Each file corresponds to a record in `proxy_records/`
with `_err` suffix.

Format: 2 columns — age [ka], 1σ uncertainty [ka].

**Uncertainty sources**:
- **GICC05**: Annual layer counting for Greenland ice cores (~0.65 kyr at 60 ka)
- **AICC2012**: Bayesian synchronization of Antarctic ice cores (~0.8–1.7 kyr)
- **LR04**: Orbital tuning of marine benthic δ¹⁸O stack (~2–4 kyr)
- **U-Th**: Uranium-thorium dating for speleothems (~0.1–0.5 kyr)
- **¹⁴C**: Radiocarbon dating for lake/marine sediments (~0.5–1.0 kyr)

### `orbital/`
- `La04_800ka.csv` — Laskar et al. (2004) orbital solution: eccentricity, obliquity,
  precession, and summer insolation at 65°N for 0–800 ka.

### Processed Data Files

#### `data_tiers_final.csv`
Record quality classification output from `scripts/01_preprocess_data.py`.

Columns:
- `Record` — Short record name
- `Type` — Archive type (Ice Core, Marine Sediment, Speleothem, Loess, Lake/Other)
- `Age_Min_ka`, `Age_Max_ka` — Age range
- `N_Points` — Number of data points
- `Resolution_Median_ka` — Median sampling resolution (kyr)
- `Coverage_10_60ka` — Fraction of target interval [10, 60] ka covered
- `Has_Event_Data` — Whether record has ≥5 points in H1 or H4 event windows
- `SNR_H1`, `SNR_H4`, `SNR_Best` — Signal-to-noise ratio in event windows
- `Trend_H1`, `Trend_H4`, `Trend_Best` — Trend significance score
- `Event_Coverage` — Coverage within event windows
- `HighFreq_Score` — High-frequency variability score
- `Resolution_Score` — Resolution quality score (0–10)
- `Composite_Score` — Weighted composite quality score
- `Age_Model_Quality` — Age model reliability assessment
- `Tier` — Final tier classification (Tier 1, 2, or 3)

**Tier criteria**:
- **Tier 1** (gold): Composite ≥ 5.0, has event data, age model = "High"
- **Tier 2** (silver): Composite ≥ 3.0, has event data
- **Tier 3** (bronze): All other valid records

#### `record_quality_score.csv`
Earlier version of record quality metrics (subset of above, simplified columns).

#### `NGRIP_trend_800ka.csv`
Pre-computed millennial-scale trend of NGRIP δ¹⁸O used as the ARC common-forcing proxy.
2 columns: `age` (ka), `trend` (standardized).

## Climate Windows Used in Analysis

| Window | Age Range (ka) | Description |
|--------|---------------|-------------|
| H1 event | 14.0–18.0 | Heinrich Stadial 1 |
| H4 event | 36.0–40.0 | Heinrich Stadial 4 |
| Bølling-Allerød | 12.7–14.7 | Warm interstadial |
| LGM | 19.0–26.0 | Last Glacial Maximum |
| GI-12 | 44.0–47.0 | Greenland Interstadial 12 |
| GS-12 | 47.0–49.0 | Greenland Stadial 12 |
| 0–400 ka | 0–400 | Early half (800-ka analysis) |
| 400–800 ka | 400–800 | Late half (800-ka analysis) |

## Regional Classification

Records are grouped into 7 regions for stacking analysis:

1. **North Atlantic** — NGRIP, ODP 980/983, U1302/1303, U1308, U1391, Fram Strait
2. **Antarctic & Southern Ocean** — EDC, Dome Fuji, WAIS, ODP 1090/1094, PS75 cores,
   U1537, MD97-2120
3. **Tropical** — Cariaco, ODP 1089/1143/1146/967, IPWP, KX97322-4, U1542, U1483,
   U1446, ODP 663, M125, Malawi, Titicaca, Ohrid
4. **East Asian Monsoon** — Cheng 2016 composite, Gulang, Weinan, Luochuan, Xifeng,
   Yimaguan, ZB13, CHB14, Jingyuan
5. **South American Monsoon** — Botuvera, Santana, Paixão, Lapa, Sudwala
6. **High-Latitude Lakes** — El'gygytgyn, Baikal, Junín
7. **Other** — Records not fitting above categories
