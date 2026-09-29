"""src/v5_code.py + src/v5_config.py → notebooks/syto16_canonical_match_v5.ipynb"""
import json, re
from pathlib import Path
root = Path(__file__).resolve().parent
code = (root / "v5_code.py").read_text()
parts = re.split(r"^# %% \[code\].*\n", code, flags=re.M)
parts = [p.strip("\n") for p in parts if p.strip()]
config = re.sub(r"^# %% \[code\].*\n", "", (root / "v5_config.py").read_text(), flags=re.M).strip("\n")

def md(s): return dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(True))
def cc(s, hide=False):
    return dict(cell_type="code", metadata={"jupyter": {"source_hidden": True}} if hide else {},
                execution_count=None, outputs=[], source=s.strip("\n").splitlines(True))

cells = [
md("""
# SYTO16 R1 → R2 canonical cell matching **v5** (whole-brain)

**목적:** R1(기준, SNR 높음)의 SYTO16 세포 목록을 canonical로 두고, R2에서 같은 세포를 1:1로 찾습니다. R2에서 검출되지 않은 세포는 주변 세포들의 이동(국소 변형)으로 위치를 추정해 보완하고, R1과 대응되지 않는 R2 검출은 제외합니다.
**최종 JSON은 입력과 같은 형식(`[[x, y, z], ...]`)이며, 행 수와 순서가 R1과 같습니다 (행 i = R1의 행 i와 같은 세포).**
관측/추정 구분 등 부가 정보는 `canonical_nodes.npz`에 따로 저장합니다. 이미지는 합성하거나 변경하지 않습니다.

**실행 순서:** 셀 0 → 셀 1(구현) → 셀 2(설정 확인) → `result = run(CFG)` → z=1980 시각화 / QC
"""),
md("""
## v4가 whole-brain에서 실패한 원인 (v4 코드와 진단 로그에서 확인한 내용)

1. **처음 10장(20 µm)만으로 R2 전체 위치를 찾는 구조.** v4는 R1 z=[1818,1828) 10장을 2D(XY 회전·배율·이동)로만 맞춘 뒤, 그 평면 주변 ±120 µm에서 R2를 찾았습니다. 두 라운드 사이에 샘플이 기울어져 있으면, R1의 얇은 한 장이 R2에서는 XY 위치마다 서로 다른 z를 지나갑니다. 그래서 일부 XY 영역에서만 맞고(band 6: 엣지 보존 0.87–0.90, p90 3.7–4.8 µm인데 `observed_pairs_too_concentrated_in_xy`로 탈락), 대부분의 블록은 대응점을 탐색 범위 끝에서 찾았습니다(`search_boundary` 약 280–306 / 360).
2. **R2 중심 z를 추측값으로 설정.** `center_z_r2=None`일 때 R2 z 범위에서 같은 비율 위치로 추측하고, 0/±50/±100/±200장만 시험했습니다. 예전 일부 z-stack 테스트는 R1·R2를 같은 z=1975–1985에서 잘라 넣었기 때문에 이 문제가 드러나지 않은 것으로 보입니다 (이 부분은 로그를 보고 추론한 것입니다).
3. **엄격한 통과/실패 기준 하나로 전체 실행이 중단됨.** band 5는 대응점 309–384쌍, 엣지 보존 0.77–0.86을 얻었지만 p90 6.06–7.09 µm가 기준 6.0 µm를 조금 넘어 탈락했습니다. 중앙 구간이 실패하면 확장 단계는 시작되지 않습니다.
4. **`propagation_progress.json`이 없는 이유:** 이 파일은 중앙 구간이 통과한 뒤 실행되는 확장 함수(`extend`) 안에서만 기록됩니다. 중앙 구간에서 실패했으므로 파일이 생성되지 않았습니다. 실행 셀의 오류 메시지가 이 파일을 확인하라고 안내한 것은 잘못된 안내였습니다.

## v5의 방법 (말씀하신 "100장씩 나눠서 처리 + 전체 refinement"의 확장판)

| 단계 | 내용 | 논의한 아이디어와의 관계 |
|---|---|---|
| 1. 전역 3D affine | 뇌 전체 점 밀도(밴드패스)의 NCC로 회전·**기울기**·배율·이동 추정. 후보: 동일 방향, 180° 뒤집힘, z축 15° 간격 회전, PCA 주축 정렬 | "뇌를 감싸는 정육면체의 축 차이 보정"을 이상치에 강한 형태로 구현 (관성 주축 + 밀도 정합) |
| 2. 독립 3D 블록 | 800 → 400 → 200 µm 블록마다 국소 이동을 따로 추정. 실패한 블록은 **중단하지 않고** 이웃 블록 값으로 채우고 이상치를 제거 | "100장씩 나눠 처리" (각 조각이 서로 독립이라 한 곳의 실패가 전체로 번지지 않음) + 전역 평활화 = refinement |
| 3. 점 기반 미세 보정 | 타일(XY 2 mm × Z 200 µm = 100장)마다 상호 최근접 쌍 + **이웃 일관성 필터**(주변 매칭쌍과 다르게 움직인 쌍 제외) → 격자 보정 2회 | 상대좌표/이웃 관계. 형률님 spectral consensus와 같은 원리(쌍들이 서로 일관된 집합만 채택)를 전체 뇌 규모에서 계산 가능한 형태로 구현 |
| 4. 최종 1:1 매칭 | 반경 5 µm 상호 최근접 → 국소 변형 보정 후 4 µm로 1회 더 매칭. 타일 경계의 중복 배정은 더 가까운 쌍만 유지 | pruning(R1과 대응되지 않는 R2 검출 제외) |
| 5. 누락 세포 보완 | R2 위치 = 전역+국소 변형 + **주변 10개 매칭 세포의 잔차 중앙값**. 추정점은 `status=2`로 구분 | "주변 B, C, D의 이동으로 A의 위치를 추정" |
| QC | kNN 엣지 보존율 vs 셔플 null (형률님 `somaprint_consistency`와 같은 방식), 단일 슬라이스 영상 근거 지수 (관측/추정/무작위 위치 비교) | 형률님 evidence index의 단순형 |

**합성 데이터 검증 (이 노트북 코드 그대로 실행):** 82만 개 세포, R2에서 20% 누락 + 10% 가짜 검출, 20° 회전 + 4° 기울기 + 400 µm 이동 + 비선형 변형(최대 ~20 µm).
동일성 정밀도 99.6%, 재현율 96.9%, 누락 세포 추정 위치 오차 중앙값 2.2 µm (p90 4.2 µm). whole-volume 모드와 빠른 테스트 모드(`R1_Z_RANGE`) 모두 통과했습니다.
**단, N407 실제 데이터에서 실행한 결과가 아니고 실제 처리 시간도 측정하지 않았습니다.** 실제 데이터에서는 아래 QC 그림과 로그 수치로 확인해 주세요.
"""),
md("""
## 환경
Python ≥ 3.8, numpy, scipy, matplotlib, tifffile (v4 환경 그대로 사용 가능). Linux 서버 (병렬 처리에 `fork` 사용).
메모리: whole-brain(점 2.4억 개) 기준으로 대략 수십 GB가 필요합니다 (v4 설정 `memory_gb=512` 환경을 가정).
"""),
cc(parts[0]),
md("## 1. 구현 (수정할 필요 없음, 접어 두어도 됩니다)"),
cc(parts[1], hide=True),
md("""
## 2. 사용자 설정

- 입력은 v4와 같은 `blobs_whole.json`입니다. 다른 파일(예: `blobs_v11.json`, `blobs_v1.json`)을 쓰시려면 `R1_JSON`, `R2_JSON`만 바꾸시면 됩니다.
- **빠른 테스트**를 원하시면 `R1_Z_RANGE=(1975, 1986)`로 설정하세요. 이 z 범위의 R1 세포만 출력하고, 정합에는 앞뒤 250장을 함께 사용합니다. whole-brain 실행은 `None`입니다.
- 출력 빈도: 단계 전환 시에 출력하고, 각 단계 안에서는 25% 단위(남은 예상 시간 포함) 또는 10분 간격(`PROGRESS_MIN_SEC`)으로만 출력합니다.
- `RESUME=True`이면 같은 설정에서 이미 끝난 단계(전역/블록/ICP)는 다시 계산하지 않습니다.
"""),
cc(config),
md("## 3. 실행"),
cc("result = run(CFG)"),
md("""
## 4. 전역 정합 확인
빨강 = R1 밀도, 초록 = 정합된 R2 밀도입니다. 두 색이 겹쳐 노랗게 보이면 전역 정합이 맞은 것입니다.
"""),
cc("""from IPython.display import Image, display
import json
display(Image(filename=str(result["run_dir"] / "global_overlay.png")))
rep = result["report"]
print(json.dumps({k: rep[k] for k in ("global", "levels", "icp")}, indent=1, ensure_ascii=False, default=str)[:4000])"""),
md("""
## 5. z=1980 단일 슬라이스 시각화 (눈으로 확인)
- 왼쪽: R1 z=1980 원본 TIFF 한 장과 그 z(±1장)의 R1 점.
- 오른쪽: **이 세포들이 R2에서 실제로 위치한 z**(중앙값)의 원본 TIFF 한 장과 최종 R2 점. 초록 = 관측(R2 검출과 매칭), 자홍 = 추정(보완), 주황 = 제외된 R2 검출.
  라운드 사이에 샘플이 기울어져 있어서, R1의 한 장에 있는 세포들이 R2에서는 여러 z에 걸쳐 있습니다. 그래서 같은 번호의 z 단면을 비교하면 안 됩니다.
- 확대 그림: 같은 번호 = 같은 R1 세포 ID입니다. R2 확대 이미지는 해당 세포 자신의 R2 z 단면입니다.
- 마지막 히스토그램: R2 영상에서 점 위치의 밝기 대비입니다. 추정점의 분포가 관측점에 가까우면 검출기가 놓친 실제 세포이고, 무작위 위치에 가까우면 R2에서 신호가 사라진 세포입니다.
"""),
cc("""vis = visualize_slice(result, CFG, z_r1=CFG["VIS_Z_R1"], dz=CFG["VIS_DZ"], crops=3)"""),
md("## 6. 최종 파일 확인"),
cc("""import numpy as np
rep = result["report"]
print("최종 R2 JSON :", rep["final_json"])
print("노드 정보    :", rep["nodes_npz"])
print(f"R1 {rep['n_r1']:,}행 = 최종 R2 {rep['n_observed'] + rep['n_imputed']:,}행")
print(f"  관측(R2 검출 매칭) {rep['n_observed']:,} / 추정 보완 {rep['n_imputed']:,}")
print("  매칭 거리(µm):", rep["match_dist_um"])
print("  그래프 엣지 보존율:", rep["edge_preservation"])
d = np.load(rep["nodes_npz"])
print("canonical_nodes.npz 키:", list(d.keys()))"""),
md("""
### 출력 파일 (`OUT_ROOT/run_<설정해시>/`)

| 파일 | 내용 |
|---|---|
| `R2_canonical_matched_to_R1.json` | **최종 결과.** 입력과 같은 형식. 행 i = R1 행 i (빠른 테스트에서는 `canonical_nodes.npz`의 `r1_row[i]`). 관측 행은 R2 원본 좌표를 그대로 사용하고, 추정 행은 예측 좌표를 사용 |
| `canonical_nodes.npz` | `r1_row`, `r1_xyz_vox`, `r2_xyz_vox`, `status`(1 관측 / 2 추정), `r2_row`(R2 원본 행 번호, 추정은 -1), `match_dist_um`, `local_mad_um`(주변 변형의 흩어짐 = 추정 불확실성), `n_support` |
| `R2_unmatched_detections.json` | R1과 대응되지 않아 제외된 R2 검출 (whole-brain 모드) |
| `warp_final.npz` | R1 µm → R2 µm 변환 (affine + 다단계 변형장). `Warp.load(path).apply(xyz_um)`로 다른 채널 점에도 적용 가능 |
| `reliability_L*.npz` | 블록별 정합 상태 코드/피크 강도 = 영역별 정합 신뢰도 지도 |
| `report.json`, `global_overlay.png`, `slice_*.png` | 요약과 QC 그림 |

**주의:** 추정점(`status=2`)은 해당 위치에 R2 신호가 있다는 뜻이 아닙니다. 이후 TACTIC이나 copositive 분석에서는 `status`를 함께 사용해 주세요.
"""),
md("""
## 7. 실패하거나 결과가 이상할 때
- 로그의 **전역 NCC가 0.5 미만**이거나 overlay가 겹치지 않으면 좌표 순서(`COORD_ORDER`), spacing, 또는 방향 뒤집힘을 확인해 주세요. 로그 첫 부분에 좌표 열과 TIFF 크기가 출력됩니다. 방향을 알고 계시면 `INITIAL_AFFINE`(4×4, R1 µm → R2 µm)을 넣어 주시면 됩니다.
- 블록 단계의 `search_boundary` 비율이 높으면 해당 level의 `search`를 키워 주세요. v5는 이 경우에도 멈추지 않고 이웃 블록 값으로 채웁니다.
- 커널이 재시작되었으면 셀 0–2를 실행한 뒤 `result = run(CFG)`를 다시 실행하세요. `RESUME=True`이면 끝난 단계는 캐시에서 읽습니다.
"""),
]
nb = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
out = root.parent / "notebooks" / "syto16_canonical_match_v5.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(out, len(cells), "cells")
