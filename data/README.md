# Place downloaded CSVs / raw datasets here.

The app caches `yfinance` data into `data/cache.db` (ignored by git). When
that cache is populated the app uses the live data; if it's missing or
yfinance is unreachable, the app falls back to a correlated synthetic panel
so the dashboard still renders.