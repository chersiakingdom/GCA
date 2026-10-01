# AAV-Cre M1 (n = 4) vs intranasal EV reinjection (n = 3)

분석 전략 문서

> 5. EV network pattern vs intranasal EV injection 시 recombination cell pattern

에 해당하는 figure 코드입니다.

**`reinj_all_in_one.py` 한 파일만 Jupyter cell 에 붙여 넣고 실행하면 됩니다.**
세 reinjection path 는 이미 들어 있습니다.

Figure 1 run 폴더(`FIG1_RUN_DIR`)는 비워 두면 자동으로 찾습니다.

1. 노트북에 남아 있는 기존 코드의 변수
   (`RESULTS`, `OVERVIEW`, `SCATTER_RESULTS`, `HELLINGER_RESULTS`,
   `RUN_DIR`, `SCATTER_RUN_DIR`, `HELLINGER_RUN_DIR`).
   `Figure_list` 경로를 들고 있어도 옆의 `run_...` 을 찾습니다.
2. `SEARCH_ROOTS` · reinjection path · 현재 폴더 · home 아래에서
   `SELECT_outputs/run_*` 중 `RUN_STATUS.json` 이 completed 인 폴더를
   탐색하여 가장 최근 것을 사용하고, 다른 후보도 함께 출력합니다.

둘 다 실패하면 어떤 변수와 어떤 폴더를 찾아봤는지 출력하므로,
`FIG1_RUN_DIR` 에 경로를 직접 적거나 `SEARCH_ROOTS` 에 상위 폴더를
추가하시면 됩니다.

Figure 포맷/스타일(Arial, SVG only, √p, source color, ρ·H 표기)은 기존
Figure 1 코드들과 동일합니다.

---

## 입력

- reinjection: 각 path 의
  `source/results/tdt_total_cell_count_{rh,whole}_lvl{1..7}.csv`
  (`lh = whole − rh`, lvl1~7 을 합친 뒤 region id 로 중복 제거)
- AAV-Cre M1: 기존 Figure 1 run 폴더의
  `config.json`, `RUN_STATUS.json`, `input_manifest.csv`,
  `raw_readout_regions.csv`, `off_source_counts.csv`,
  `predefined_recipient_regions.csv`, `source_region_definition.csv`,
  `primary/AllGrayMatter/<source>/feature_definitions.csv`

출력은 `<run_dir>/../Figure_list_reinjection` 에 SVG + 값 CSV 로 저장됩니다.

---

## Hemisphere (문서 4번)

left / right 를 합치지 않고 각각 독립 feature 로 유지합니다.
한 mouse 의 p 는 `2 × region` 개 feature 전체에서 합이 1 입니다.

- AAV-Cre: injection side 기준 ipsilateral / contralateral 로 정렬
- intranasal reinjection: injection hemisphere 가 없으므로 RH / LH 를
  같은 두 자리에 둠 (slot 1 ↔ Cre ipsilateral, slot 2 ↔ Cre contralateral)

이 대응은 intranasal 에서 임의 선택이므로, 뒤집은 경우
(`ORIENTATION = "rh_to_contra"`)의 Hellinger 평균도 함께 계산해서 CSV 에
남기고 Hellinger figure 에 점선으로 표시합니다.

`HEMI_MODE = "pooled"` 로 두면 양쪽을 합친 bilateral 분석도 가능합니다
(보조용).

---

## 생성되는 figure

| 파일 | 내용 | 근거 |
|---|---|---|
| `Reinj_overview_heatmap_<scope>.svg` | √p heatmap. Cre(ipsi/contra) + reinjection(RH/LH), 개체 + mean | III-1, 4 |
| `Reinj_vs_CreM1_scatter.svg` | Cre M1 reference vs reinjection scatter, hemisphere 별 marker, ρ·H | III-3, 8 |
| `Reinj_Hellinger_reference_comparison.svg` | reinjection → Cre M1 reference 를 Cre LOO · Cre pairwise(natural inter-animal variability)와 같은 축에 표시 | 7, 9, III-2 |
| `Reinj_source_reference_specificity.svg` | reinjection × 4 source reference heatmap + ΔH margin | 11, 5-A |
| `Reinj_recipient_region_enrichment.svg` | M1 recipient region enrichment vs constrained permutation (RH/LH 각각) | 5-B, 10 |
| `Reinj_reference_density_model.svg` | reference density 가 reinjection count 를 예측하는지 (NB2, hemisphere 포함) | 5-C |
| `Reinj_QC_signal_coverage_laterality.svg` | 절대 signal, coverage, 분포 집중도, laterality | 5 Step 0, IV-2 |

Scope 는 `PredefinedM1` / `Predefined20` / `AllGrayMatter` 세 가지입니다.
source-specificity 에서는 region 자체가 M1 에서 정의된 `PredefinedM1` 을
circularity 때문에 제외합니다.

---

## 그 밖의 처리 원칙

- **Normalization**: mouse 별 `count / area` 를 sum = 1 로 normalize.
  Eq6 의 off-source denominator 는 한 mouse 안에서 상수라 p 에서
  소거되므로 두 dataset 의 area 단위가 달라도 영향이 없습니다.
  절대 cell 수는 QC figure 에서 따로 다룹니다(IV-2).
- **Source region**: 네 source 의 union(및 descendant)을 양쪽에서 동일하게
  제외합니다(3번, cf Step 1).
- **Parent-child overlap**: broad pool 은 자신의 descendant 가 pool 에 함께
  있지 않은 region 만 남겨 구성합니다(2번).
- **Permutation**: p-value 는 "major division 과 region volume 만 맞춘
  random region set 보다 집중이 높은가"에 대한 값이며, biological
  replicate 수준의 동일성을 뜻하지 않습니다(10번, IV-4).
- **Negative control**: vehicle / naive plasma / EV-depleted plasma 등
  데이터가 코드에 없으므로 5번 Step 0 의 "signal > negative control" 은
  수행되지 않고, QC figure 는 두 group 의 절대량 비교까지입니다.

---

## 참고

`_synthetic_check.py` 는 실제 데이터 없이 합성 데이터로 코드가 끝까지
도는지 확인하는 개발용 스크립트입니다(`python _synthetic_check.py <폴더>`).
분석에는 필요하지 않습니다.
