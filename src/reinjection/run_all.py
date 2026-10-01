# ============================================================
# AAV-Cre M1 (n=4) vs intranasal EV reinjection (n=3)
#
# 전체 figure 를 순서대로 생성합니다.
#
# 사용법 (notebook):
#
#   import sys
#   sys.path.append("/path/to/GCA/src/reinjection")
#   import reinj_config as C
#   C.FIG1_RUN_DIR = RESULTS["output"]   # 또는 직접 경로
#   from run_all import run_all
#   REINJ = run_all()
#
# 사용법 (shell):
#
#   python run_all.py --fig1-run-dir /path/to/run_... --no-show
# ============================================================

import argparse
import traceback

try:
    from . import reinj_common as K
    from . import reinj_config as C
except ImportError:
    import reinj_common as K
    import reinj_config as C


STEPS = [
    ("overview", "reinj_fig01_overview", "make_overview"),
    ("scatter", "reinj_fig02_scatter", "make_scatter"),
    ("hellinger", "reinj_fig03_hellinger", "make_hellinger"),
    ("source_specificity",
     "reinj_fig04_source_specificity", "make_source_specificity"),
    ("enrichment", "reinj_fig05_enrichment", "make_enrichment"),
    ("reference_model",
     "reinj_fig06_reference_density_model", "make_reference_model"),
    ("qc", "reinj_fig07_qc", "make_qc"),
]


def _load(module_name, function_name):

    try:
        module = __import__(
            f"{__package__}.{module_name}",
            fromlist=[function_name],
        )
    except (ImportError, ValueError, TypeError):
        module = __import__(module_name, fromlist=[function_name])

    return getattr(module, function_name)


def run_all(only=None, stop_on_error=False):

    ctx = K.load_fig1()

    tables = K.load_reinj_tables()

    print()
    print("==========================================")
    print("INPUT")
    print("==========================================")
    print("Figure 1 run:")
    print(ctx["run"])
    print()
    print("출력 폴더:")
    print(ctx["output"])
    print()
    print(f"AAV-Cre {C.SOURCE_LABEL[C.MATCHED_SOURCE]} mice "
          f"(n = {len(K.cre_m1_mice(ctx))}):")
    for mouse in K.cre_m1_mice(ctx):
        print(f"  {mouse}")
    print()
    print(f"Reinjection mice (n = {len(tables)}):")
    for mouse, frame in tables.items():
        print(f"  {mouse}: {len(frame)} regions")
    print()
    print(f"Hemisphere mode: {C.HEMI_MODE}")

    results = {}

    for name, module_name, function_name in STEPS:

        if only is not None and name not in only:
            continue

        print()
        print("##########################################")
        print(f"# {name}")
        print("##########################################")

        function = _load(module_name, function_name)

        try:
            results[name] = function(ctx=ctx, tables=tables)
        except Exception:  # noqa: BLE001
            if stop_on_error:
                raise
            print()
            print(f"[{name}] 실패:")
            traceback.print_exc()
            results[name] = None

    print()
    print("==========================================")
    print("ALL DONE")
    print("==========================================")
    print("저장 폴더:")
    print(ctx["output"])
    print()

    for name, value in results.items():
        print(f"{name}: {'ok' if value is not None else 'FAILED'}")

    return dict(ctx=ctx, tables=tables, results=results)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--fig1-run-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hemi-mode", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    args = parser.parse_args()

    if args.fig1_run_dir:
        C.FIG1_RUN_DIR = args.fig1_run_dir

    if args.output_dir:
        C.OUTPUT_DIR = args.output_dir

    if args.hemi_mode:
        C.HEMI_MODE = args.hemi_mode

    if args.no_show:
        C.SHOW_FIGURES = False

    run_all(only=args.only, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
