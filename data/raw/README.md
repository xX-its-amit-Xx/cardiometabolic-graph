# `data/raw/` — drop external datasets here

Everything in this directory is **gitignored** (see `.gitignore`). Layout the
loaders expect:

```
data/raw/
├── mimic-iv/                  # MIMIC-IV demo OR full credentialed
│   ├── hosp/
│   │   ├── patients.csv.gz
│   │   ├── admissions.csv.gz
│   │   ├── labevents.csv.gz
│   │   ├── d_labitems.csv.gz
│   │   └── prescriptions.csv.gz
│   └── icu/
│       └── chartevents.csv.gz
├── nhanes-2017-18/            # NHANES 2017-18 cycle XPT files
│   ├── DEMO_J.XPT
│   ├── PAQ_J.XPT
│   ├── DR1TOT_J.XPT
│   ├── SLQ_J.XPT
│   └── SMQ_J.XPT
└── reactome/                  # Reactome flat files (Homo sapiens)
    ├── ReactomePathways.txt
    ├── ReactomePathwaysRelation.txt
    ├── NCBI2Reactome.txt
    └── ChEBI2Reactome.txt
```

If a directory is missing, the corresponding loader logs a warning and
continues — the pipeline still produces a runnable demo using only the
synthetic engagement data.

## Where to download

- **MIMIC-IV demo (no credentials)** — https://physionet.org/content/mimic-iv-demo/2.2/
- **MIMIC-IV full (credentialed)** — https://physionet.org/content/mimiciv/
- **NHANES 2017-18** — https://wwwn.cdc.gov/Nchs/Nhanes/continuousnhanes/default.aspx?BeginYear=2017
- **Reactome** — https://reactome.org/download-data
