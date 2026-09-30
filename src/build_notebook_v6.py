"""src/v6_code.py + src/v6_config.py → notebooks/syto16_canonical_match_v6.ipynb"""
import json, re
from pathlib import Path
root = Path(__file__).resolve().parent
code = (root / "v6_code.py").read_text()
parts = re.split(r"^# %% \[code\].*\n", code, flags=re.M)
parts = [p.strip("\n") for p in parts if p.strip()]
config = re.sub(r"^# %% \[code\].*\n", "", (root / "v6_config.py").read_text(), flags=re.M).strip("\n")

def md(s): return dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(True))
def cc(s, hide=False):
    return dict(cell_type="code", metadata={"jupyter": {"source_hidden": True}} if hide else {},
                execution_count=None, outputs=[], source=s.strip("\n").splitlines(True))

cells = [
md("""
# SYTO16 R1 → R2 canonical cell matching **v6** — 상대좌표 그래프 매칭

**목적:** R1(기준)의 세포 목록을 canonical graph로 두고, R2에서 각 세포의 위치를 찾거나(관측) 주변 세포들의 이동으로 추정해(보완) **R1과 같은 노드·같은 순서의 R2 점 목록**을 만듭니다. 이미지는 합성하거나 바꾸지 않습니다.
**최종 JSON은 입력과 같은 형식 `[[x, y, z], ...]`이며 행 수·순서가 R1과 같습니다.** 관측/보완 구분과 신뢰도는 `canonical_nodes.npz`에 저장합니다.

**실행 순서:** 셀 0 → 셀 1(구현) → 셀 2(설정) → `result = run(CFG)` → 확인/시각화
기본 설정은 v5 결과의 변형장(`INIT_WARP`)을 출발점으로 써서 1~3단계(약 1시간)를 건너뜁니다.
"""),
md("""
## v5 결과를 신뢰하기 어려운 이유 (합성 데이터로 확인)

실제 데이터 v5 결과: 관측 21%, 매칭 거리 p50 2.9 µm, 엣지 보존율 0.88 (null 0.003).
정답을 아는 합성 데이터에서 실제 데이터처럼 **블록 정합이 많이 실패하는 조건**을 만들면, v5는 **같은 모양의 QC 수치**를 보였습니다(매칭 거리 p50 3.25 µm, 엣지 보존율 0.87 vs null 0.007). 그런데 그 매칭의 **98%가 다른 세포**였습니다. 즉 v5의 QC는 매칭이 맞는지를 가려내지 못합니다.

**원인:** v5는 변형장으로 옮긴 절대좌표에서 가장 가까운 R2 점을 짝으로 택했습니다. 변형장이 수십 µm 틀린 곳에서도 5 µm 안에 우연히 있는 다른 세포를 짝으로 택하고, 그 짝으로 다시 변형장을 맞추는 순환이 생겼습니다.

## v6의 방법 (단계별)

| 단계 | 하는 일 | 논의 아이디어와의 대응 |
|---|---|---|
| 0 | JSON 읽기, µm 변환, 공간 색인. 뇌를 타일(XY 2 mm × Z 100장)로 나눔 | "나눠서 처리" |
| 1–3 | 전역 affine → 블록 변형 → ICP (v5와 동일). 기본값은 v5 결과 재사용 | "뇌 전체 정육면체 축 보정" |
| 4a | **이웃 합의 투표:** 세포 i의 후보 R2 변위를 주변 R1 이웃 24개도 똑같이 갖는지 셈. 진짜 국소 이동은 이웃이 공유하고, 우연한 짝은 공유하지 않음. 넓은 범위(35 → 18 µm)에서 변형장의 틀린 곳을 바로잡음 | 형률님 spectral matching(서로 일관된 대응끼리 강화) |
| 4b/5 | **상대좌표 그래프 매칭 (전체 세포):** 모든 세포를 "이웃 10개까지의 상대 벡터 집합"으로 표현. R1 세포와 R2 후보의 벡터 집합이 얼마나 겹치는지(0~1)로 짝을 정함. 이동에 무관하고, 회전·기울기는 변형장으로 보정한 상대 벡터로 비교. 누락·가짜 이웃이 있어도 집합 비교라 견고 | 다예님 relative coordinate + local graph relationship |
| 5 | **점수 보정 (FDR):** 같은 계산을 무작위 위치(40 µm 이동)에서도 해서 우연히 나오는 점수 분포(null)를 구하고, 거짓 매칭률 1% 이하가 되는 점수를 자동 임계값으로 사용. 노드별 신뢰도 = 1 − local FDR | 형률님 확률적 판단(연속 점수) |
| 5 | **2차 매칭:** 그래프로 확정된 이웃들의 국소 변형으로 위치를 보정한 뒤, 남은 세포를 4 µm 상호 최근접으로 연결 | 보완 후 재매칭 |
| 5 | **같은 R1 평면 우선:** 이웃을 찾을 때 z 차이를 5배로 계산. 같은 R1 z-평면의 세포들이 R2에서도 하나의 매끄러운 면을 이루도록 함(절대 z가 아니라 상대 관계). 그 면에서 벗어난 짝은 제외 | 교수님 요청 |
| 5 | **누락 세포 보완:** 주변 관측 세포 12개(80 µm 이내)로 가중 국소 affine(TPS/ARAP의 국소 선형판)을 맞춰 위치와 불확실성(µm)을 산출 | "주변 B, C, D의 이동으로 A 추정" |
| 검증 | **hold-out:** 관측 세포 5%를 가리고 주변으로 맞힌 오차. **영상 근거:** R1 세포 patch와 R2 위치 patch의 NCC, 밝기 대비 | 형률님 evidence index / hold-out |

**노드 상태 (`status`)**

| 값 | 의미 |
|---|---|
| 1 | 관측 — R2 검출과 매칭 (`match_method` 1 = 그래프, 2 = 국소 보정 후 2차 매칭) |
| 2 | 보완 — 주변 관측 세포 6개 이상으로 국소 변형을 추정 (hold-out 정확도가 이 경우에 해당) |
| 3 | 보완 — 주변 근거 부족, 전역 변형장만 사용. **신뢰 낮음.** 행은 유지하지만 후속 분석에서 구분 필요 |

**합성 데이터 검증** (정답을 아는 82만 세포; R2 누락 20–35%, 가짜 검출 10–20%, 회전 20°, 기울기 4°, 비선형 변형)

| 조건 | 방법 | 관측 정확도 (precision) | 재현율 | 보완 위치 오차 (중앙값) |
|---|---|---|---|---|
| 중간 변형 | v5 | 99.6% | 96.9% | 2.2 µm |
| 중간 변형 | v6 | 99.6% | 95.7% | status 2: 2.5 µm (99%가 5 µm 이내) |
| 강한 국소 변형 (블록 정합 대부분 실패) | v5 | **36%** | 9% | 33 µm (구분 없음) |
| 강한 국소 변형 | v6 | **97%** | 24% | status 2: 3.8 µm / status 3: 45 µm (신뢰 낮음으로 표시) |

v6는 변형장이 틀린 곳에서 **틀린 짝을 만들지 않고 "모른다"(status 3)로 표시**합니다. 실제 데이터가 어느 조건에 가까운지는 실행 후 로그의 FDR·hold-out·status 비율로 알 수 있습니다.
**이 수치는 합성 데이터 결과이며, N407 실제 데이터로는 아직 실행하지 않았습니다.**
"""),
md("""
## 환경
Python ≥ 3.8, numpy, scipy, matplotlib, tifffile. Linux 서버 (병렬 처리에 `fork` 사용). whole-brain(2.4억 점) 기준 메모리 수십 GB.
"""),
cc(parts[0]),
md("## 1. 구현 (수정할 필요 없음, 접어 두어도 됩니다)"),
cc(parts[1], hide=True),
md("""
## 2. 사용자 설정
- `INIT_WARP`: v5 실행의 `warp_final.npz`를 출발점으로 사용합니다(1~3단계 생략). 경로가 다르면 수정하시고, 처음부터 하시려면 `None`으로 두세요.
- 빠른 확인: `R1_Z_RANGE=(1975, 1986)`. 이 z 범위의 R1만 출력하고, 정합에는 앞뒤 250장을 함께 씁니다.
- 출력: 단계 전환 시, 그리고 단계 안에서는 25% 단위 또는 10분 간격으로만 출력합니다.
- `RESUME=True`: 같은 설정으로 끝난 단계는 다시 계산하지 않습니다. 단계마다 설정 해시를 따로 두므로, 뒤 단계 설정만 바꾸면 그 단계부터 다시 계산합니다.
"""),
cc(config),
md("## 3. 실행"),
cc("result = run(CFG)"),
md("""
## 4. 요약과 검증 그림
- `graph_est_fdr`: 채택한 그래프 매칭의 추정 거짓 매칭률(null 기반).
- `holdout`: 관측 세포를 가리고 맞힌 오차. status 2의 정확도를 나타냅니다.
- `holdout_error_map.png`: 영역별 hold-out 오차. 빨간 곳은 보완 위치가 덜 정확한 영역입니다.
"""),
cc("""from IPython.display import Image, display
import json
rep = result["report"]
keys = ["n_r1", "n_observed", "n_graph_matched", "n_nn_filled", "n_imputed_supported", "n_imputed_unsupported",
        "graph_threshold", "graph_est_fdr", "holdout", "imputed_uncertainty_um", "graph_passes"]
print(json.dumps({k: rep.get(k) for k in keys}, indent=1, ensure_ascii=False, default=str))
p = result["run_dir"] / "holdout_error_map.png"
if p.exists():
    display(Image(filename=str(p)))"""),
md("""
## 5. z=1980 단일 슬라이스 시각화
- 왼쪽: R1 z=1980 원본 TIFF와 그 z(±1장)의 R1 점(청록).
- 오른쪽: 그 세포들이 R2에서 위치한 z(중앙값)의 원본 TIFF와 최종 점. 초록 = 관측, 자홍 = 보완(근거 충분), 파랑 = 보완(근거 부족), 주황 = 제외된 R2 검출.
- 확대 그림: 노란 번호가 같으면 같은 R1 세포입니다. R2 확대는 그 세포 자신의 R2 z 단면입니다.
- 영상 근거 히스토그램: (a) R2 밝기 대비, (b) R1 세포 patch와 R2 위치 patch의 NCC. 회색(무작위/어긋난 위치)보다 오른쪽에 있을수록 실제 신호와 맞습니다.
"""),
cc("""vis = visualize_slice(result, CFG, z_r1=CFG["VIS_Z_R1"], dz=CFG["VIS_DZ"], crops=3)"""),
md("""
## 6. 출력 파일 (`OUT_ROOT/run_<해시>/`)

| 파일 | 내용 |
|---|---|
| `R2_canonical_matched_to_R1.json` | **최종 결과.** 입력과 같은 형식, 행 i = R1 행 i. 관측 행은 R2 원본 좌표, 보완 행은 추정 좌표 |
| `canonical_nodes.npz` | `status`(1/2/3), `match_method`, `graph_score`, `match_confidence`(관측: 1 − local FDR), `imputed_uncertainty_um`, `n_support`, `r2_row`, 좌표 |
| `R2_unmatched_detections.json` | R1과 대응되지 않아 제외된 R2 검출 |
| `warp_final.npz` | R1 µm → R2 µm 변환. `Warp.load(path).apply(xyz_um)` |
| `holdout_error_map.png`, `slice_*.png`, `report.json` | 검증 그림과 요약 |
"""),
md("""
## 7. 추가 진단 (선택)
관측 비율의 공간 분포, 예측 위치 → 가장 가까운 R2까지의 거리(우연 수준과 비교), 블록 정합 신뢰도 지도를 그립니다.
블록 지도는 1~3단계를 이 실행에서 계산했을 때만 있습니다(`INIT_WARP` 사용 시 v5 실행 폴더에 있음).
"""),
cc(re.sub(r"^# %% \[code\].*\n", "", (root / "v5_diagnostics.py").read_text(), flags=re.M).strip("\n")),
]
nb = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
out = root.parent / "notebooks" / "syto16_canonical_match_v6.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(out, len(cells), "cells")
