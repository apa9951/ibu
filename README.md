# ibu — Insights on Breast-screening Users
*ibu* means "mother" in Indonesian.

FRE 6191 final project · Dash dashboard on 6.79 M screening mammograms (BCSC Risk Factors dataset, 2005–2017).

## Run locally
```bash
pip install -r requirements.txt
python app.py          # open http://127.0.0.1:8050
```
The three CSVs in `data/` are read directly (relative paths), so the app runs without internet.

## Files
| File | What it is |
|---|---|
| `app.py` | Dash app: data prep, 16 charts, 12 callbacks |
| `assets/style.css`, `assets/logo.svg` | ibu theme and logo (Dash loads `assets/` automatically) |
| `data/bcsc_risk_factors_summarized{1,2,3}_092020.csv` | BCSC data, one dataset cut into 3 year ranges |
| `requirements.txt` | Python dependencies (includes `gunicorn` for Render) |

## Story
1 Prevention → 2 Detection → 3 Screening (density) → 4 Risk markers → 5 Survivors → 6 Business segments & persona builder → 7 Takeaways.

## Data
Breast Cancer Surveillance Consortium Risk Factors Dataset (summarized, 09/2020) — https://www.bcsc-research.org/datasets/rf
Supported by the National Cancer Institute (P01CA154292). One row = a combination of risk factors; `count` = number of exams.

## Deploy on Render (free)
1. Push this folder to a public GitHub repo (keep `data/` inside it; each CSV is 15 MB, under GitHub's limits).
2. Render → New → Web Service → connect the repo.
3. Build command `pip install -r requirements.txt` · Start command
   `gunicorn app:server --workers 1 --timeout 120 --bind 0.0.0.0:$PORT` · Instance type Free.
   (`render.yaml` holds the same settings for New → Blueprint. `.python-version` pins Python 3.13.)
4. Memory: about 280 MB with the full page loaded, under the free 512 MB. Keep 1 worker.
5. The free tier sleeps after 15 min idle; the first visit after that takes about a minute.

## v2 additions
- Landing page (survival-by-stage bars, 1-in-8, count-up stats; sources: American Cancer Society 2026, WHO 2024).
- Centered top menu with LinkedIn and GitHub icons; years slider appears after the landing page.
- Chapter 2 **Geography**: U.S. state map (hand-drawn, works offline), state profile tiles, state-level scatter.
  Data: `data/us_state_context.csv`, `data/us_states.geojson` (built by `prepare_state_data.py`, source in file header).
- Contact section; scroll animations respect `prefers-reduced-motion`.

## Notebook
- `ibu_notebook.ipynb` — exploration and thought process (32 charts incl. 11 state maps; every dashboard chart has its exploratory version here, 5 research questions with hypotheses). Run from this folder.
- `ibu_notebook.html` — the same notebook with all outputs, opens in any browser.
- `build_notebook.py` — regenerates the notebook.
