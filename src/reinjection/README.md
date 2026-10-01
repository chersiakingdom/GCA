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
   `run_*` 폴더를 탐색합니다.

찾은 run 은 **폴더 이름이 아니라 내용으로** 검증합니다.

- `RUN_STATUS.json` 이 completed
- 필요한 CSV 7개가 모두 존재
- cohort 의 `Cre` group 에 M1 / dHP / S1 / PFC 가 모두 존재

조건을 만족하는 run 이 하나면 그것을 쓰고(어떤 cohort 구성인지 함께 출력),
여러 개면 **자동으로 고르지 않고 멈춘 뒤** 후보 목록을 출력합니다.
GFP-hue 처럼 source 구성이 다른 분석의 run 은 제외되며 이유가 출력됩니다.

`FIG1_RUN_DIR` 을 직접 지정하신 경우에도 같은 검증을 거칩니다.

### 문제가 생기면

```python
diagnose()
```

run 폴더 후보와 각각의 cohort 구성·제외 사유, 선택된 run 의 CSV column,
reinjection lvl 파일의 column 과 row 수, region id 매칭 상태를 출력합니다.
figure 는 만들지 않습니다.

Figure 포맷/스타일(Arial, SVG only, √p, source color, ρ·H 표기)은 기존
Figure 1 코드들과 동일합니다.

---

## 전처리는 Figure 1 분석 코드와 동일합니다

reinjection CSV 는 Figure 1 분석 코드(`run_analysis`)의
`load_one_csv` / `merge_levels` / `load_count_family` 와 같은 규칙으로
읽습니다.

- column: `id`, `region`, `count`, `area`
- `ex_co/results` → `results_cocheck` → `results` 우선순위로 폴더 선택
- whole/rh × lvl1~7 (행이 없는 level 파일은 건너뛰고 그 사실을 출력)
- 여러 level 에 같은 ID 가 있으면 값이 일치할 때만 1회 사용
- `lh = whole − rh`, 음수는 허용 오차 밖이면 중단

feature 는 **새로 정의하지 않고 Figure 1 run 의 것을 그대로 읽어
reinjection 에 적용**합니다.

- `primary/Predefined/Motor/feature_definitions.csv` + `p.csv`
- `primary/AllGrayMatter/Motor/feature_definitions.csv` + `p.csv`
- `source_specificity/shared_feature_definitions.csv` +
  `p_shared_feature_space.csv`

즉 `parent − nearest included descendants` residual, source 제외,
zero-volume feature 제외가 모두 Figure 1 과 동일하게 적용되며,
AAV-Cre 쪽 p 는 Figure 1 이 계산해 둔 값을 그대로 씁니다.
reinjection 은 같은 정의로 count/area 를 만들어
`p = (count/area) / Σ(count/area)` 로 normalize 합니다
(Eq6 의 배율과 denominator 는 p 에서 상쇄됩니다).

---

## 입력

- reinjection: 각 path 아래
  `source/results/tdt_total_cell_count_{whole,rh}_lvl{1..7}.csv`
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

Scope 는 Figure 1 run 의 `Predefined` 와 `AllGrayMatter` 이며,
source-specificity 는 네 source 의 union 을 공통 제외한
`source_specificity/` feature space 를 사용합니다.

---

## 그 밖의 처리 원칙

- **Normalization / source 제외 / parent-child overlap**: 모두 Figure 1
  feature 정의를 그대로 따릅니다(위 참조).
- **절대 cell 수**는 pattern 과 분리해 QC figure 에서 다룹니다(IV-2).
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
