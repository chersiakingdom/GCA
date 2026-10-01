# ============================================================
# 합성 데이터 smoke test
#
# 실제 데이터가 없는 환경에서, Figure 1 분석 코드(run_analysis)가
# 만드는 것과 같은 구조의 run 폴더와 reinjection CSV 를 만들어
# reinj_all_in_one.py 가 끝까지 도는지 확인합니다.
#
#   python _synthetic_check.py [작업폴더]
#
# 분석에는 필요하지 않은 개발용 스크립트입니다.
# ============================================================

from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd


RNG = np.random.default_rng(7)

SOURCES = ["Motor", "dHP", "S1", "PFC"]

DIVISIONS = [
    ("Isocortex", 315),
    ("TH", 549),
    ("HPF", 1089),
    ("STR", 477),
    ("MB", 313),
]

GRAY_ROOTS = [2, 639, 1014]


def build_atlas():
    """root -> gray roots -> division -> region -> subregion"""

    rows = [
        dict(id=997, name="root", acronym="root",
             parent_structure_id=-1, depth=0),
    ]

    for rid, name in zip(GRAY_ROOTS, ["Cerebrum", "Brain stem", "Cerebellum"]):
        rows.append(dict(id=rid, name=name, acronym=name.replace(" ", ""),
                         parent_structure_id=997, depth=1))

    for index, (acronym, rid) in enumerate(DIVISIONS):
        rows.append(dict(id=rid, name=acronym, acronym=acronym,
                         parent_structure_id=GRAY_ROOTS[index % 3], depth=2))

    region_id = 10000
    child_id = 20000

    for acronym, division_id in DIVISIONS:

        for index in range(12):

            rows.append(dict(
                id=region_id, name=f"{acronym} region {index}",
                acronym=f"{acronym}{index}",
                parent_structure_id=division_id, depth=3,
            ))

            # 일부 region 에는 child 를 둬서 residual feature 가 생기게 함
            if index % 4 == 0:

                rows.append(dict(
                    id=child_id, name=f"{acronym} region {index} part a",
                    acronym=f"{acronym}{index}a",
                    parent_structure_id=region_id, depth=4,
                ))

                child_id += 1

            region_id += 1

    return pd.DataFrame(rows)


def build(root):

    root = Path(root)

    if root.exists():
        shutil.rmtree(root)

    run = root / "EVnetwork_Cre_SELECT_outputs" / "run_synthetic"

    run.mkdir(parents=True)

    atlas = build_atlas()

    atlas_path = root / "AllBrainRegions.csv"

    atlas.to_csv(atlas_path, index=False)

    parents = {
        int(r.id): (None if r.parent_structure_id < 0
                    else int(r.parent_structure_id))
        for r in atlas.itertuples()
    }

    def ancestors(rid):
        chain, node = [], parents[rid]
        while node is not None:
            chain.append(node)
            node = parents[node]
        return chain

    measured = [
        int(r.id) for r in atlas.itertuples()
        if r.depth >= 2 or int(r.id) in GRAY_ROOTS
    ]

    leaves = [
        rid for rid in measured
        if not any(parents.get(other) == rid for other in measured)
    ]

    # ----------------------------------------------------
    # source / predefined
    # ----------------------------------------------------

    source_ids = {}

    for index, source in enumerate(SOURCES):
        source_ids[source] = leaves[index * 2: index * 2 + 2]

    used = {rid for ids in source_ids.values() for rid in ids}

    recipients = [rid for rid in leaves if rid not in used]

    predefined = {}

    for index, source in enumerate(SOURCES):
        predefined[source] = recipients[index * 5: index * 5 + 5]

    pd.DataFrame([
        dict(source=source, id=rid, name=atlas.set_index("id").at[rid, "name"],
             minimal_source_root=True)
        for source, ids in source_ids.items() for rid in ids
    ]).to_csv(run / "source_region_definition.csv", index=False)

    pd.DataFrame([
        dict(source=source, acronym=atlas.set_index("id").at[rid, "acronym"],
             id=rid, name=atlas.set_index("id").at[rid, "name"])
        for source, ids in predefined.items() for rid in ids
    ]).to_csv(run / "predefined_recipient_regions.csv", index=False)

    # ----------------------------------------------------
    # cohort / manifest
    # ----------------------------------------------------

    cohorts = {"Cre": {}, "SELECT": {}}

    manifest_rows = []

    for source in SOURCES:

        for group, n in (("Cre", 4), ("SELECT", 3)):

            mice = [f"{group}_{source}_{i}" for i in range(1, n + 1)]

            cohorts[group][source] = mice

            for position, mouse in enumerate(mice):
                manifest_rows.append(dict(
                    mouse=mouse, group=group, source=source,
                    injection_side="R" if position % 2 == 0 else "L",
                ))

    manifest = pd.DataFrame(manifest_rows)

    manifest.to_csv(run / "input_manifest.csv", index=False)

    (run / "config.json").write_text(json.dumps({
        "config": {
            "atlas_path": str(atlas_path),
            "area_atol": 1e-6,
            "count_atol": 1e-8,
            "eq6_scale": 1e6,
        },
        "cohorts": cohorts,
        "source_ids": source_ids,
        "predefined": {s: [str(i) for i in ids]
                       for s, ids in predefined.items()},
    }, indent=2), encoding="utf-8")

    (run / "RUN_STATUS.json").write_text(
        json.dumps({"status": "completed", "output": str(run)}),
        encoding="utf-8",
    )

    # ----------------------------------------------------
    # source 별 pattern / region 크기
    # ----------------------------------------------------

    area = pd.Series(
        RNG.lognormal(10.0, 0.7, size=len(measured)), index=measured
    )

    patterns = {}

    for source in SOURCES:

        weight = pd.Series(
            RNG.lognormal(0.0, 0.6, size=len(leaves)), index=leaves
        )

        weight.loc[predefined[source]] *= 9.0

        patterns[source] = weight / weight.sum()

    # ----------------------------------------------------
    # mouse 별 raw count (leaf 에 신호를 주고 ancestor 로 합산)
    # ----------------------------------------------------

    def rollup(leaf_counts):

        values = pd.Series(0.0, index=measured)

        for rid, value in leaf_counts.items():

            values[rid] += value

            for ancestor in ancestors(rid):
                if ancestor in values.index:
                    values[ancestor] += value

        return values

    raw_rows = []

    for group in ("Cre", "SELECT"):

        for source in SOURCES:

            for mouse in cohorts[group][source]:

                side = manifest.loc[
                    manifest.mouse == mouse, "injection_side"
                ].iloc[0]

                total = int(RNG.integers(9000, 20000))

                noisy = patterns[source] * RNG.lognormal(
                    0.0, 0.25, size=len(leaves)
                )

                noisy = noisy / noisy.sum()

                ipsi = rollup(pd.Series(
                    RNG.poisson(noisy.to_numpy() * total * 0.75),
                    index=leaves,
                ))

                contra = rollup(pd.Series(
                    RNG.poisson(noisy.to_numpy() * total * 0.25),
                    index=leaves,
                ))

                ipsi_key = "rh" if side == "R" else "lh"
                contra_key = "lh" if side == "R" else "rh"

                for rid in measured:

                    raw_rows.append({
                        "mouse": mouse,
                        "id": rid,
                        "region": atlas.set_index("id").at[rid, "name"],
                        f"{ipsi_key}_count": float(ipsi[rid]),
                        f"{contra_key}_count": float(contra[rid]),
                        f"{ipsi_key}_area": float(area[rid] / 2),
                        f"{contra_key}_area": float(area[rid] / 2),
                        "whole_count": float(ipsi[rid] + contra[rid]),
                        "whole_area": float(area[rid]),
                    })

    raw = pd.DataFrame(raw_rows)

    raw.to_csv(run / "raw_readout_regions.csv", index=False)

    raw_by_mouse = {m: f.set_index("id") for m, f in raw.groupby("mouse")}

    pd.DataFrame([
        dict(mouse=mouse, group="Cre", source="Motor",
             off_source_readout_count=float(frame["whole_count"].sum()))
        for mouse, frame in raw_by_mouse.items()
    ]).to_csv(run / "off_source_counts.csv", index=False)

    # ----------------------------------------------------
    # feature space (partition) 와 p 저장
    # ----------------------------------------------------

    def build_features(wanted, excluded):

        boundaries = set(wanted) | set(excluded)

        children = {rid: [] for rid in boundaries}

        for child in boundaries:
            for parent in ancestors(child):
                if parent in boundaries:
                    children[parent].append(child)
                    break

        records = []

        for hemi in ("ipsi", "contra"):

            drop = set(excluded) if hemi == "ipsi" else set()

            for rid in sorted(wanted):

                if rid in drop or (set(ancestors(rid)) & drop):
                    continue

                subtract = sorted(
                    c for c in children[rid] if c not in drop
                )

                records.append(dict(
                    feature=f"{hemi}::{rid}",
                    hemisphere=hemi,
                    region_id=rid,
                    region_name=atlas.set_index("id").at[rid, "name"],
                    region_label=atlas.set_index("id").at[rid, "acronym"]
                    + (" [residual]" if subtract else ""),
                    subtract_ids=subtract,
                    subtract_names=[],
                    label=f"{hemi} | {rid}",
                ))

        return pd.DataFrame(records).set_index("feature")

    def values_for(mouse, features, metric):

        frame = raw_by_mouse[mouse]

        side = manifest.loc[
            manifest.mouse == mouse, "injection_side"
        ].iloc[0]

        out = []

        for record in features.itertuples():

            key = (
                ("rh" if side == "R" else "lh")
                if record.hemisphere == "ipsi"
                else ("lh" if side == "R" else "rh")
            )

            column = f"{key}_{metric}"

            value = float(frame.at[record.region_id, column])

            if record.subtract_ids:
                value -= float(
                    frame.loc[record.subtract_ids, column].sum()
                )

            out.append(max(value, 0.0))

        return np.array(out)

    gray = [
        rid for rid in measured
        if any(rid == g or g in ancestors(rid) for g in GRAY_ROOTS)
    ]

    def save_space(directory, mice, wanted, excluded):

        directory.mkdir(parents=True, exist_ok=True)

        features = build_features(wanted, excluded)

        count = pd.DataFrame(
            [values_for(m, features, "count") for m in mice],
            index=mice, columns=features.index,
        )

        area_frame = pd.DataFrame(
            [values_for(m, features, "area") for m in mice],
            index=mice, columns=features.index,
        )

        keep = features.index[(area_frame > 1e-6).all(axis=0)]

        count, area_frame = count[keep], area_frame[keep]

        features = features.loc[keep]

        density = count / area_frame

        p = density.div(density.sum(axis=1), axis=0)

        count.rename_axis("mouse").to_csv(directory / "count.csv")
        area_frame.rename_axis("mouse").to_csv(directory / "area.csv")
        p.rename_axis("mouse").to_csv(directory / "p.csv")

        saved = features.copy()
        saved["subtract_ids"] = saved["subtract_ids"].map(
            lambda x: json.dumps(x)
        )
        saved["subtract_names"] = saved["subtract_names"].map(
            lambda x: json.dumps(x)
        )
        saved["excluded_zero_area"] = False
        saved.to_csv(directory / "feature_definitions.csv")

        return features

    for source in SOURCES:

        mice = cohorts["Cre"][source] + cohorts["SELECT"][source]

        save_space(
            run / "primary" / "AllGrayMatter" / source,
            mice, gray, source_ids[source],
        )

        save_space(
            run / "primary" / "Predefined" / source,
            mice, predefined[source], source_ids[source],
        )

    union = sorted({rid for ids in source_ids.values() for rid in ids})

    everyone = [m for s in SOURCES for g in ("Cre", "SELECT")
                for m in cohorts[g][s]]

    spec_dir = run / "source_specificity"

    features = save_space(spec_dir, everyone, gray, union)

    (spec_dir / "p.csv").rename(spec_dir / "p_shared_feature_space.csv")

    (spec_dir / "feature_definitions.csv").rename(
        spec_dir / "shared_feature_definitions.csv"
    )

    # ----------------------------------------------------
    # reinjection CSV (id, region, count, area)
    # ----------------------------------------------------

    reinj_roots = {}

    for number in range(1, 4):

        mouse_root = root / f"reinj_{number}"

        results = mouse_root / "source" / "results"

        results.mkdir(parents=True)

        reinj_roots[f"Reinj_{number}"] = str(mouse_root)

        total = int(RNG.integers(800, 2500))

        noisy = patterns["Motor"] * RNG.lognormal(
            0.0, 0.8, size=len(leaves)
        )

        noisy = noisy / noisy.sum()

        whole_leaf = pd.Series(
            RNG.poisson(noisy.to_numpy() * total), index=leaves
        ).astype(float)

        rh_leaf = pd.Series(
            RNG.binomial(whole_leaf.astype(int), 0.5), index=leaves
        ).astype(float)

        whole = rollup(whole_leaf)
        rh = rollup(rh_leaf)

        level_of = {rid: min(max(int(d), 1), 7)
                    for rid, d in zip(atlas.id, atlas.depth)}

        for hemi, counts, areas in (
            ("whole", whole, area),
            ("rh", rh, area / 2),
        ):

            for level in range(1, 8):

                ids = [
                    rid for rid in measured
                    if level_of[int(rid)] == level
                ]

                frame = pd.DataFrame({
                    "id": ids,
                    "region": [
                        atlas.set_index("id").at[rid, "name"] for rid in ids
                    ],
                    "count": [float(counts[rid]) for rid in ids],
                    "area": [float(areas[rid]) for rid in ids],
                })

                if frame.empty:
                    frame = pd.DataFrame(
                        columns=["id", "region", "count", "area"]
                    )

                frame.to_csv(
                    results
                    / f"tdt_total_cell_count_{hemi}_lvl{level}.csv",
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
