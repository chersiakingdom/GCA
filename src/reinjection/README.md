# AAV-Cre M1 (n = 4) vs intranasal EV reinjection (n = 3)

분석 전략 문서의

> 5. EV network pattern vs intranasal EV injection 시 recombination cell pattern

에 해당하는 figure 생성 코드입니다. Figure 포맷/스타일(Arial, SVG only,
source color, √p heatmap, ρ·H 표기 등)은 기존 Figure 1 코드들과 동일하게
맞추었습니다.

---

## 1. 입력

### Reinjection (n = 3)

`reinj_config.REINJ_ROOTS` 의 각 path 아래

```
source/results/tdt_total_cell_count_rh_lvl{1..7}.csv
source/results/tdt_total_cell_count_whole_lvl{1..7}.csv
```

- `lh = whole − rh` 로 계산합니다.
- lvl1~7 을 합친 뒤 region id 기준으로 중복을 제거하며, 같은 region 이
  여러 level 파일에서 다른 값을 가지면 에러를 냅니다.

### AAV-Cre M1 (n = 4)

기존 Figure 1 분석 폴더(`RESULTS["output"]`)의

```
config.json, RUN_STATUS.json, input_manifest.csv,
raw_readout_regions.csv, off_source_counts.csv,
predefined_recipient_regions.csv, source_region_definition.csv,
primary/AllGrayMatter/<source>/feature_definitions.csv
```

를 그대로 사용합니다. Cre cohort 중 `Motor` source 의 mouse 가 비교 대상입니다.

---

## 2. 실행

Notebook:

```python
import sys
sys.path.append("/path/to/GCA/src/reinjection")

import reinj_config as C
C.FIG1_RUN_DIR = RESULTS["output"]      # 또는 직접 경로
C.REINJ_ROOTS = {                        # 기본값이 이미 들어 있습니다
    "Reinj_1": "/data5/20231202_14_23_01_2nd_EV_reinj_#4_IN_M1_1_destriped_DONE",
    "Reinj_2": "/data5/20240104_15_44_33_2nd_EV_reinj_#5_Intranasal_M1_2_destriped_DONE",
    "Reinj_3": "/data5/20231203_13_15_02_2nd_EV_reinj_#6_IN_M1_3_destriped_DONE",
}

from run_all import run_all
REINJ = run_all()
```

Shell:

```bash
python run_all.py --fig1-run-dir /path/to/run_... --no-show
```

개별 figure 만 다시 그리려면 `run_all(only=["hellinger"])` 처럼 지정하거나
해당 파일을 단독 실행하면 됩니다.

출력 폴더는 기본적으로 `<run_dir>/../Figure_list_reinjection` 입니다.

---

## 3. 생성되는 figure

| 파일 | 내용 | 근거 |
|---|---|---|
| `Reinj_overview_heatmap_<scope>.svg` | √p regional distribution heatmap. 왼쪽 AAV-Cre M1 개체 + mean, 오른쪽 reinjection 개체 + mean | III-1 |
| `Reinj_vs_CreM1_scatter.svg` | Cre M1 reference(√p) vs reinjection(√p) scatter, Spearman ρ 와 Hellinger 표기 | III-3, 8 |
| `Reinj_Hellinger_reference_comparison.svg` | reinjection → Cre M1 reference 를 Cre M1 LOO 및 Cre M1 pairwise(natural inter-animal variability)와 같은 축에 표시 | 7, 9, III-2 |
| `Reinj_source_reference_specificity.svg` | reinjection mouse × 4 source reference Hellinger heatmap + margin bar | 11, 5-A optional |
| `Reinj_recipient_region_enrichment.svg` | Figure 1 M1 recipient region 으로의 enrichment 와 constrained permutation null | 5-B, 10 |
| `Reinj_reference_density_model.svg` | Figure 1 reference density 가 reinjection regional count 를 예측하는지(NB2 count model) | 5-C |
| `Reinj_QC_signal_coverage_laterality.svg` | 절대 signal, coverage, distribution 집중도, laterality | 5 Step 0, 4, IV-2 |

값은 모두 같은 폴더에 CSV 로 함께 저장됩니다.

---

## 4. 분석 설계상 유의한 점

### 4-1. Hemisphere

Intranasal 투여에는 injection hemisphere 가 없어 ipsilateral /
contralateral 을 정의할 수 없습니다. 따라서 primary analysis
(`HEMI_MODE = "pooled"`)에서는 **양쪽 group 모두 bilateral** 로 맞춥니다.

- AAV-Cre: ipsi + contra
- Reinjection: rh + lh (= whole)

RH 를 ipsilateral 로 간주하는 것은 생물학적 근거가 없으므로 기본값이
아니며, sensitivity analysis 용으로 `HEMI_MODE = "rh_as_ipsi"` 만
제공합니다. Cre 의 ipsilateral 편향 자체는 QC figure 의 laterality panel
에서 reinjection 과 비교해 볼 수 있습니다.

### 4-2. Normalization

mouse 별 `density = count / area` 를 구한 뒤 sum = 1 로 normalize 하여 p 를
만듭니다. Eq6 의 off-source denominator 는 한 mouse 안에서 모든 region 에
동일하게 작용하는 상수라 p 계산에서 소거되므로, 두 dataset 의 area 단위가
달라도 p 는 영향을 받지 않습니다. 절대 cell 수는 QC figure 에서 별도로
다룹니다(IV-2).

### 4-3. Source region

네 source region 의 union(및 descendant)을 양쪽 dataset 에서 모두
제외합니다. source-specificity 분석에서 reference 마다 서로 다른 region 을
빼면 distance 를 직접 비교할 수 없기 때문입니다(cf 절 Step 1).

### 4-4. Parent-child overlap

broad pool 은 atlas hierarchy 상 자신의 descendant 가 pool 안에 함께
존재하지 않는 region 만 남겨 구성합니다(2번).

### 4-5. Permutation 의 의미

`Reinj_recipient_region_enrichment` 의 p-value 는

> 실제 recipient region 으로의 집중이 major division 과 region volume 만
> 맞춘 random region set 보다 높은가

에 대한 값이며, biological replicate 수준의 동일성을 뜻하지 않습니다
(10번, IV-4).

### 4-6. Negative control

이 코드에는 vehicle / naive plasma / EV-depleted plasma 등 negative control
데이터가 들어 있지 않습니다. 따라서 5번 Step 0 의 "signal > negative
control" 확인은 수행되지 않으며, QC figure 는 두 group 의 절대량을
기술적으로 비교하는 수준입니다. Negative control 이 준비되면 같은 형식으로
추가할 수 있습니다.

---

## 5. 실제 데이터에서 먼저 확인이 필요한 두 가지

이 코드는 `/data5` 에 접근할 수 없는 환경에서 작성되어, 합성 데이터로
end-to-end 동작만 검증했습니다(`python _synthetic_check.py <폴더>`).
따라서 실제 파일에 대해 아래 두 가지는 첫 실행 시 확인해 주시면 좋겠습니다.

1. **lvl CSV 의 column 이름**
   region id / cell count / region area 에 해당하는 column 을 자동으로
   찾도록 되어 있고(`reinj_common.py` 의 `_ID_CANDIDATES` 등),
   찾지 못하면 실제 column 목록을 포함한 에러를 냅니다. 이름이 다르면 해당
   후보 목록에 한 줄 추가하면 됩니다.

2. **`feature_definitions.csv` 에 residual 정의가 있는지**
   AllGrayMatter feature 가 `parent − children` 형태의 residual 로 정의되어
   있으면, 같은 정의를 reinjection data 에도 적용해야 합니다. 그런
   column 이 감지되면 조용히 넘어가지 않고 에러를 내도록 해 두었습니다.
   이 경우 해당 column 형식을 알려 주시면 그 부분만 보완하겠습니다.
   그때까지는 `SCOPES` 에서 `AllGrayMatter` 를 빼고
   predefined region 분석만 수행할 수 있습니다.
