# RUBiVAR — Rubin variable-star classification in early LSST data

Classification of variable stars from Vera C. Rubin Observatory / LSST data, with
**RR Lyrae** as the priority class: what can and cannot be done with the few epochs the
early data releases provide.

Method: a 5-class gradient-boosted classifier on classical light-curve features, scored on
held-out sky fields, benchmarked against a transformer on raw light curves (DL) and a
semi-supervised VAE (generative) under one protocol (`dp2/bench/`).

---

## Repository layout

```
RUBiVAR/
├── dp1/        # DP1 — archivio congelato (pipeline completa, codice + dati + risultati)
├── dp2/        # DP2 — lavoro attivo
│   └── catalogs/   # cataloghi di crossmatch Gaia DR3 × LSST DP2
├── requirements.txt
└── LICENSE
```

Ogni release ha il proprio albero autoconsistente: script, dati e risultati stanno insieme.
`BASE` negli script è derivato da `__file__`, quindi ogni cartella è eseguibile in loco senza
path assoluti.

---

## DP1 — archivio (`dp1/`)

Prima iterazione completa su LSST DP1: 13.505 curve di luce `strict`, crossmatch Gaia DR3
(SOS + match posizionale 2″ + classifier), 392 oggetti etichettati, SS-VAE a 32 dimensioni,
classificatore binario RR e multiclasse sul latente.

Vedi `dp1/README.md` per la pipeline, i risultati e lo stato preciso in cui il lavoro si è
fermato. La cartella è di sola consultazione: non modificarla.

---

## DP2 — lavoro attivo (`dp2/`)

Nuova iterazione sui dati DP2, con un ground truth molto più ampio: **~34.000 stelle LSST
classificate come variabili da Gaia DR3**.

Cataloghi di partenza (in `dp2/catalogs/`):

| File | Ruolo |
|---|---|
| `lsst_dp2_gaia_sos.csv` | Gaia SOS — 34.553 oggetti, 11 flag `in_vari_*`; classi ad alta affidabilità, sorgente primaria |
| `lsst_dp2_gaia_classifier_result.csv` | Gaia classifier — 33.485 oggetti con `best_class_name` / `best_class_score`; fallback per chi non ha classe SOS |
| `class_definition.txt` | Legenda delle 24 classi del classificatore Gaia |

Copertura: 25.584 oggetti hanno almeno un flag SOS attivo (898 ne hanno più di uno, da
risolvere per priorità); i restanti 8.969 possono prendere la classe dal classifier.

### Struttura

```
dp2/
├── main.py            entry point unico
├── config.py          path e iperparametri — unica fonte di verità
├── catalog.py         ground truth Gaia (SOS → classifier) + campi di cielo e fold
├── lightcurves.py     Lomb-Scargle + phase-fold
├── features.py        feature classiche + mappe dm–dt
├── classify.py        classificatore a 5 classi + candidate RR, valutato su campi tenuti fuori
├── bench/             benchmark ML (HGB) / DL (transformer) / generativo (SS-VAE)
├── slurm/             job per Leonardo (CINECA)
├── catalogs/          cataloghi Gaia + ground truth + periodi + feature
└── rsp/                gira sulla Rubin Science Platform
    ├── crossmatch_gaia.py    crossmatch Gaia DR3 × DP2 → cataloghi in catalogs/
    └── download_lc.py        curve di luce → lightcurves/
```

```bash
python main.py catalog                  # ground truth Gaia + campi di cielo
python main.py fold --min-points 30     # Lomb-Scargle + phase-fold
python main.py features                 # feature classiche + dm–dt
python main.py classify                # classificatore, campi tenuti fuori
python main.py all                      # i quattro stadi in sequenza

# benchmark, stesso protocollo per i tre modelli (M = hgb | transformer | vae)
python main.py bench-cache                         # sequenze + feature sulle prime n notti
python main.py bench --model M --step tune         # Optuna sui campi di validazione interni
python main.py bench --model M --step eval         # test 1-2: campi tenuti fuori, prime n notti
python main.py bench --model M --step lc           # test 3: curva di apprendimento
python main.py bench --step report                 # tabella unica → results/bench/report.txt
```

Lo scaricamento delle curve di luce **non** è uno stadio di `main.py`: gira su RSP,
l'unico posto in cui esiste `lsst.rsp`.

### Pipeline

```
catalog   ground truth: SOS prioritario, classifier come fallback;
          oggetti raggruppati in campi di cielo (DBSCAN), campi interi divisi in 5 fold
  ↓
fold      Lomb-Scargle multibanda + phase-fold
  ↓
features  120 feature classiche (ampiezze, forma della distribuzione, Stetson, χ²,
          Fourier, colori, periodo) + mappe dm–dt a Δt grossolano
  ↓
classify  gradient boosting a 5 classi (RR, ECL, LPV, AGN, OTHER = ogni altra
          etichetta), valutato out-of-fold su campi mai visti. Due prodotti:
          classe più probabile con flag di bassa confidenza, e candidate RR
          con P(RR) sopra soglia (scelta sugli altri fold); risultati anche per
          numero di notti osservate
```

### Perché valutare su campi tenuti fuori, e perché le feature classiche

I campi DP2 hanno cadenze e composizioni di classi molto diverse (i due campi nel piano
galattico contengono quasi tutte le LPV, alcuni campi extragalattici quasi tutti gli AGN).
Con uno split casuale ogni campo sta sia nel training sia nel test, e un modello può
riconoscere il campo invece della stella: le sole feature di campionamento davano AUC-PR
0.41 su RR vs ECL con lo split casuale e 0.13 (sotto il caso) su campi tenuti fuori.

Su campi tenuti fuori (AUC-PR, RR vs ECL / RR vs resto):

| Ingresso | RR vs ECL | RR vs resto |
|---|---|---|
| latente SS-VAE, riaddestrato per fold (vecchio ramo, sostituito da `bench/vae.py`) | 0.24 | 0.32 |
| mappe dm–dt | 0.70 | 0.63 |
| feature classiche | 0.83 | 0.70 |
| **feature classiche + dm–dt** | **0.85** | **0.74** |

La classe OTHER serve perché nel catalogo vero ci sono anche le altre classi: senza,
finivano quasi tutte in ECL (precisione ECL 0.50) e 236 in RR. Con OTHER la precisione
sale a 0.79 (ECL) e 0.73 (RR), e P(RR) del modello a 5 classi batte un binario RR vs resto
dedicato (AUC-PR 0.80 contro 0.75). Una testa per ogni classe con ≥100 etichette peggiora
invece le RR (0.73): RS, YSO, ELL e CV hanno etichette solo dal classificatore Gaia.

Il latente dipende dalla curva piegata, ma con il campionamento DP2 il periodo
Lomb-Scargle è corretto solo per ~30% delle RR: su campi con poche notti la curva piegata
è rumore e la rappresentazione non si trasferisce. Le feature classiche e dm–dt usano
informazioni che non richiedono il periodo giusto.

## Setup

```bash
pip install -r requirements.txt
```

Training su GPU NVIDIA (CUDA). Il phase-folding e Lomb-Scargle usano multiprocessing su CPU.

---

## Note sui dati

Nessun dato è versionato (vedi `.gitignore`): curve di luce, fold, periodi, feature,
cataloghi, modelli e risultati restano fuori dal repository. I dati DP2 sono soggetti ai
diritti di accesso Rubin, quindi "riproducibile" significa: rilanciabile da chi ha accesso
a DP2 sulla Rubin Science Platform, nell'ordine

    rsp/crossmatch_gaia.py → rsp/download_lc.py      (su RSP)
    main.py all → main.py bench-cache → main.py bench ...

Ogni risultato del benchmark salva commit git, impronte dei file di input e versioni dei
pacchetti (`results/bench/<modello>/meta_*.json`). `_admin/` contiene materiale non
attinente al progetto ed è esclusa dal versionamento.
