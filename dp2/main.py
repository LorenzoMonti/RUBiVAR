"""
RUBiVAR DP2 — single entry point.

    python main.py catalog                 build the Gaia ground truth
    (download the light curves on RSP: rsp/download_lc.py)
    python main.py fields                  sky fields from shared visits + field folds
    python main.py fold  [--min-points N]  Lomb-Scargle + phase-fold
    python main.py features                classical features + dm-dt maps
    python main.py classify                5-class classifier + RR candidates, held-out fields
    python main.py all                     the five stages above, in order
    python main.py period-check            LS period of the RR vs Gaia DR3, by nights

Benchmark ML / DL / generative, same protocol for every model (see bench/):

    python main.py bench-cache                         token sequences + first-n-nights features
    python main.py bench --model M --step tune [--folds 0 1] [--trials N]
    python main.py bench --model M --step eval [--seeds 0]   tests 1-2, held-out folds
    python main.py bench --model M --step lc                 test 3, learning curve
    python main.py bench --step report [--model M ...] [--out report.txt]
    python main.py bench --step figures [--model M ...] [--out figures]

Downloading the light curves is NOT a stage here: it runs on the Rubin Science
Platform, which is the only place `lsst.rsp` exists. See rsp/download_lc.py.

Every path and hyperparameter lives in config.py.
"""

import argparse
import os
import sys
import time


def _banner(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def stage_catalog(args):
    import catalog
    _banner("CATALOG — Gaia ground truth")
    catalog.build(min_clf_score=args.min_score)


def stage_fields(args):
    import catalog
    _banner("FIELDS — objects sharing visits, field folds")
    catalog.build_fields()


def stage_fold(args):
    import lightcurves
    _banner("FOLD — Lomb-Scargle + phase-fold")
    lightcurves.run(min_points=args.min_points, min_per_band=args.min_per_band,
                    workers=args.workers, force=args.force)


def stage_features(args):
    import features
    _banner("FEATURES — classical features + dm-dt")
    features.run(workers=args.workers)


def stage_classify(args):
    import classify
    _banner("CLASSIFY — classes + RR candidates, held-out fields")
    classify.run(no_plot=args.no_plot)


BENCH_MODELS = ("hgb", "hgb_classic", "hgb_dmdt", "transformer", "vae", "vae_hgb", "mlp_dmdt")


def stage_bench_cache(args):
    from bench import degrade, sequences
    _banner("BENCH-CACHE — token sequences, first-n-nights features")
    sequences.build()
    degrade.build()


def stage_bench(args):
    import importlib
    import config as C
    from bench import report, run
    if args.step in ("report", "figures"):
        names = args.model or [m for m in BENCH_MODELS
                               if os.path.isdir(os.path.join(C.BENCH_DIR, m))]
        _banner(f"BENCH — {args.step}")
        if args.step == "report":
            report.report(names, out=args.out or "report.txt")
        else:
            from bench import figures
            figures.figures(names, out=args.out or "figures")
        return
    if not args.model or len(args.model) != 1:
        raise ValueError("--step tune / eval takes exactly one --model")
    module = importlib.import_module(f"bench.{args.model[0]}")
    _banner(f"BENCH — {args.step} {module.NAME}")
    if args.step == "tune":
        for k in args.folds:
            run.tune(module, k, args.trials or C.BENCH_TRIALS)
    elif args.step == "eval":
        run.evaluate(module, seeds=args.seeds or C.BENCH_SEEDS, folds=args.folds)
    else:
        run.learning_curve(module, folds=args.folds)


def stage_period_check(args):
    import period_check
    _banner("PERIOD-CHECK — LS period vs Gaia DR3")
    period_check.run()


def stage_all(args):
    stage_catalog(args)
    stage_fields(args)
    stage_fold(args)
    stage_features(args)
    stage_classify(args)


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("catalog", help="build the Gaia ground truth")
    c.add_argument("--min-score", type=float, default=None,
                   help="minimum Gaia classifier score for fallback labels")
    c.set_defaults(func=stage_catalog)

    fl = sub.add_parser("fields", help="sky fields from shared visits + field folds")
    fl.set_defaults(func=stage_fields)

    f = sub.add_parser("fold", help="Lomb-Scargle + phase-fold")
    f.add_argument("--min-points", type=int, default=None,
                   help="minimum detections across all bands")
    f.add_argument("--min-per-band", type=int, default=None,
                   help="minimum detections for a band to be folded")
    f.add_argument("--workers", type=int, default=None, help="CPU processes")
    f.add_argument("--force", action="store_true", help="refold everything, ignore resume")
    f.set_defaults(func=stage_fold)

    e = sub.add_parser("features", help="classical features + dm-dt maps")
    e.add_argument("--workers", type=int, default=None, help="CPU processes")
    e.set_defaults(func=stage_features)

    k = sub.add_parser("classify", help="5-class classifier + RR candidates, held-out fields")
    k.add_argument("--no-plot", action="store_true")
    k.set_defaults(func=stage_classify)

    pc = sub.add_parser("period-check", help="LS period of the RR vs Gaia DR3, by nights")
    pc.set_defaults(func=stage_period_check)

    bc = sub.add_parser("bench-cache", help="night-cut feature caches for the benchmark")
    bc.set_defaults(func=stage_bench_cache)

    b = sub.add_parser("bench", help="tune / score / report the benchmarked models")
    b.add_argument("--model", nargs="+", choices=BENCH_MODELS)
    b.add_argument("--step", choices=["tune", "eval", "lc", "report", "figures"], required=True)
    b.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    b.add_argument("--trials", type=int, default=None, help="default C.BENCH_TRIALS")
    b.add_argument("--seeds", type=int, nargs="+", default=None, help="default C.BENCH_SEEDS")
    b.add_argument("--out", default=None,
                   help="report file / figure folder in C.BENCH_DIR (report.txt / figures)")
    b.set_defaults(func=stage_bench)

    a = sub.add_parser("all", help="catalog → fields → fold → features → classify")
    a.add_argument("--min-score", type=float, default=None)
    a.add_argument("--min-points", type=int, default=None)
    a.add_argument("--min-per-band", type=int, default=None)
    a.add_argument("--workers", type=int, default=None)
    a.add_argument("--force", action="store_true")
    a.add_argument("--no-plot", action="store_true")
    a.set_defaults(func=stage_all)

    return p


def main():
    args = build_parser().parse_args()
    t0 = time.time()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
