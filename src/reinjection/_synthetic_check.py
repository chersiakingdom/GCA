# ============================================================
# 합성 데이터 smoke test
#
# 실제 /data5 데이터가 없는 환경에서 코드가 끝까지 실행되는지,
# 그리고 입출력 형식이 서로 맞는지 확인하기 위한 스크립트입니다.
#
#   python _synthetic_check.py [작업폴더]
#
# 여기서 만드는 합성 파일의 column 이름은 "가정"입니다.
# 실제 서버 파일의 column 이름이 다르면 reinj_common.py 의
# 후보 목록(_ID_CANDIDATES 등)에 실제 이름을 추가하십시오.
# ============================================================

from pathlib import Path
import json
import sys
import shutil

import numpy as np
import pandas as pd


RNG = np.random.default_rng(7)


# ------------------------------------------------------------
# 작은 atlas
# ------------------------------------------------------------

DIVISIONS = [
    ("Isocortex", 315),
    ("TH", 549),
    ("HPF", 1089),
    ("STR", 477),
    ("MB", 313),
]


def build_atlas():

    rows = [
        dict(id=997, name="root", acronym="root", parent_structure_id=-1),
        dict(id=8, name="grey matter", acronym="grey",
             parent_structure_id=997),
    ]

    for acronym, rid in DIVISIONS:
        rows.append(dict(
            id=rid, name=acronym, acronym=acronym,
            parent_structure_id=8,
        ))

    leaf_id = 10000

    for acronym, division_id in DIVISIONS:

        for index in range(14):

            rows.append(dict(
                id=leaf_id,
                name=f"{acronym} region {index}",
                acronym=f"{acronym}{index}",
                parent_structure_id=division_id,
            ))

            leaf_id += 1

    return pd.DataFrame(rows)


def build(root):

    root = Path(root)

    if root.exists():
        shutil.rmtree(root)

    run = root / "SELECT_outputs" / "run_synthetic"

    run.mkdir(parents=True)

    atlas = build_atlas()

    atlas_path = root / "atlas.csv"

    atlas.to_csv(atlas_path, index=False)

    leaves = atlas.loc[
        atlas.parent_structure_id.isin([rid for _, rid in DIVISIONS])
    ].copy()

    # ----------------------------------------------------
    # source region (4 source × 2 region)
    # ----------------------------------------------------

    source_rows = []

    sources = ["Motor", "dHP", "S1", "PFC"]

    source_ids = []

    for index, source in enumerate(sources):

        chosen = leaves.iloc[index * 2: index * 2 + 2]

        source_ids.extend(chosen.id.tolist())

        for row in chosen.itertuples():
            source_rows.append(dict(
                id=int(row.id),
                acronym=row.acronym,
                name=row.name,
                source=source,
            ))

    pd.DataFrame(source_rows).to_csv(
        run / "source_region_definition.csv", index=False
    )

    # ----------------------------------------------------
    # predefined recipient regions (source 별 5개, 총 20개)
    # ----------------------------------------------------

    recipient_pool = leaves.loc[~leaves.id.isin(source_ids)].reset_index(
        drop=True
    )

    preset_rows = []

    for index, source in enumerate(sources):

        chosen = recipient_pool.iloc[index * 5: index * 5 + 5]

        for row in chosen.itertuples():
            preset_rows.append(dict(
                source=source,
                id=int(row.id),
                acronym=row.acronym,
                name=row.name,
            ))

    pd.DataFrame(preset_rows).to_csv(
        run / "predefined_recipient_regions.csv", index=False
    )

    # ----------------------------------------------------
    # cohort
    # ----------------------------------------------------

    cohorts = {"Cre": {}, "SELECT": {}}

    manifest_rows = []

    for index, source in enumerate(sources):

        for group in ("Cre", "SELECT"):

            mice = [
                f"{group}_{source}_{number}"
                for number in range(1, 5)
            ]

            cohorts[group][source] = mice

            for position, mouse in enumerate(mice):

                manifest_rows.append(dict(
                    mouse=mouse,
                    group=group,
                    source=source,
                    injection_side="R" if position % 2 == 0 else "L",
                ))

    manifest = pd.DataFrame(manifest_rows)

    manifest.to_csv(run / "input_manifest.csv", index=False)

    (run / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "atlas_path": str(atlas_path),
                    "area_atol": 1e-6,
                    "count_atol": 1e-8,
                },
                "cohorts": cohorts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (run / "RUN_STATUS.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    # ----------------------------------------------------
    # source 별 "진짜" regional pattern
    # ----------------------------------------------------

    analyzed = leaves.loc[~leaves.id.isin(source_ids)].copy()

    area = pd.Series(
        RNG.lognormal(mean=10.0, sigma=0.7, size=len(analyzed)),
        index=analyzed.id.to_numpy(),
    )

    patterns = {}

    for source in sources:

        preferred = set(
            r["id"] for r in preset_rows if r["source"] == source
        )

        weight = RNG.lognormal(0.0, 0.6, size=len(analyzed))

        weight = pd.Series(weight, index=analyzed.id.to_numpy())

        weight.loc[list(preferred)] *= 9.0

        patterns[source] = weight / weight.sum()

    # ----------------------------------------------------
    # raw_readout_regions.csv
    # ----------------------------------------------------

    raw_rows = []

    off_rows = []

    for group in ("Cre", "SELECT"):

        for source in sources:

            for mouse in cohorts[group][source]:

                side = manifest.loc[
                    manifest.mouse == mouse, "injection_side"
                ].iloc[0]

                total = int(RNG.integers(9000, 20000))

                noisy = patterns[source] * RNG.lognormal(
                    0.0, 0.25, size=len(analyzed)
                )

                noisy = noisy / noisy.sum()

                ipsi_counts = RNG.poisson(noisy.to_numpy() * total * 0.75)

                contra_counts = RNG.poisson(noisy.to_numpy() * total * 0.25)

                ipsi_key = "rh" if side == "R" else "lh"
                contra_key = "lh" if side == "R" else "rh"

                for position, rid in enumerate(analyzed.id.to_numpy()):

                    values = {
                        f"{ipsi_key}_count": float(ipsi_counts[position]),
                        f"{contra_key}_count": float(contra_counts[position]),
                        f"{ipsi_key}_area": float(area.loc[rid] / 2),
                        f"{contra_key}_area": float(area.loc[rid] / 2),
                    }

                    raw_rows.append(dict(
                        mouse=mouse,
                        id=int(rid),
                        acronym=analyzed.loc[
                            analyzed.id == rid, "acronym"
                        ].iloc[0],
                        **values,
                    ))

                off_rows.append(dict(
                    mouse=mouse,
                    off_source_readout_count=float(
                        ipsi_counts.sum() + contra_counts.sum()
                    ),
                ))

    pd.DataFrame(raw_rows).to_csv(
        run / "raw_readout_regions.csv", index=False
    )

    pd.DataFrame(off_rows).to_csv(
        run / "off_source_counts.csv", index=False
    )

    # ----------------------------------------------------
    # AllGrayMatter feature definitions
    # ----------------------------------------------------

    for source in sources:

        directory = run / "primary" / "AllGrayMatter" / source

        directory.mkdir(parents=True)

        rows = []

        for hemisphere in ("ipsi", "contra"):

            for row in analyzed.itertuples():

                rows.append(dict(
                    feature=f"{hemisphere}::{int(row.id)}",
                    hemisphere=hemisphere,
                    region_id=int(row.id),
                    region_name=row.name,
                    region_label=row.acronym,
                ))

        pd.DataFrame(rows).to_csv(
            directory / "feature_definitions.csv", index=False
        )

    # ----------------------------------------------------
    # reinjection lvl1~7 파일
    #
    # Motor pattern 을 기반으로 하되 noise 를 더 크게 주고
    # 좌우 대칭으로 만듭니다 (intranasal).
    # ----------------------------------------------------

    level_of = {997: 1, 8: 2}

    for _, rid in DIVISIONS:
        level_of[rid] = 3

    for rid in analyzed.id.to_numpy():
        level_of[int(rid)] = 4

    for rid in source_ids:
        level_of[int(rid)] = 4

    reinj_roots = {}

    for number in range(1, 4):

        mouse_root = root / f"reinj_{number}"

        results = mouse_root / "source" / "results"

        results.mkdir(parents=True)

        reinj_roots[f"Reinj_{number}"] = str(mouse_root)

        total = int(RNG.integers(800, 2500))

        noisy = patterns["Motor"] * RNG.lognormal(
            0.0, 0.8, size=len(analyzed)
        )

        noisy = noisy / noisy.sum()

        whole_counts = pd.Series(
            RNG.poisson(noisy.to_numpy() * total),
            index=analyzed.id.to_numpy(),
        ).astype(float)

        rh_counts = pd.Series(
            RNG.binomial(whole_counts.astype(int), 0.5),
            index=whole_counts.index,
        ).astype(float)

        # source region 과 상위 region 도 파일에는 존재
        extra_ids = list(source_ids) + [rid for _, rid in DIVISIONS] + [8, 997]

        frames = {level: [] for level in range(1, 8)}

        for rid in list(analyzed.id.to_numpy()) + extra_ids:

            rid = int(rid)

            row = atlas.loc[atlas.id == rid].iloc[0]

            if rid in whole_counts.index:
                whole_count = float(whole_counts.loc[rid])
                rh_count = float(rh_counts.loc[rid])
                whole_area = float(area.loc[rid])
            else:
                whole_count = float(RNG.integers(0, 40))
                rh_count = float(RNG.integers(0, int(whole_count) + 1))
                whole_area = float(RNG.lognormal(11.5, 0.5))

            level = level_of[rid]

            frames[level].append(dict(
                id=rid,
                name=row["name"],
                acronym=row["acronym"],
                whole_count=whole_count,
                rh_count=rh_count,
                whole_area=whole_area,
            ))

        for level in range(1, 8):

            table = pd.DataFrame(frames[level])

            if table.empty:
                table = pd.DataFrame(
                    columns=["id", "name", "acronym", "count", "area"]
                )

                table.to_csv(
                    results / f"tdt_total_cell_count_rh_lvl{level}.csv",
                    index=False,
                )

                table.to_csv(
                    results / f"tdt_total_cell_count_whole_lvl{level}.csv",
                    index=False,
                )

                continue

            table.assign(
                count=table.rh_count,
                area=table.whole_area / 2,
            )[["id", "name", "acronym", "count", "area"]].to_csv(
                results / f"tdt_total_cell_count_rh_lvl{level}.csv",
                index=False,
            )

            table.assign(
                count=table.whole_count,
                area=table.whole_area,
            )[["id", "name", "acronym", "count", "area"]].to_csv(
                results / f"tdt_total_cell_count_whole_lvl{level}.csv",
                index=False,
            )

    return run, reinj_roots


def main():

    target = sys.argv[1] if len(sys.argv) > 1 else "./_synthetic_reinj"

    run, reinj_roots = build(target)

    sys.path.insert(0, str(Path(__file__).parent))

    import matplotlib
    matplotlib.use("Agg")

    import reinj_all_in_one as R

    R.FIG1_RUN_DIR = str(run)
    R.REINJ_ROOTS = reinj_roots
    R.SHOW_FIGURES = False
    R.N_PERMUTATION = 300

    R.run_all(stop_on_error=True)


if __name__ == "__main__":
    main()
