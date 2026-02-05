# 퍼스널 컬러 립 추천 및 시뮬레이션 시스템 기술 백서

**Version**: 1.0  
**Last Updated**: 2026-01-19

---

## 1. 시스템 개요 (System Overview)

본 시스템은 사용자의 얼굴 이미지를 분석하여 **퍼스널 컬러 타입을 진단**하고, 이에 최적화된 **립 컬러를 추천**한 뒤, 사용자가 선택한 **질감(Texture)을 반영하여 가상 시뮬레이션**을 수행하는 통합 솔루션입니다.

### 1.1 핵심 프로세스
1.  **Diagnosis (진단)**: 사용자 피부 톤(LAB) 분석 → 퍼스널 컬러 타입 확정(1종) + 추천 립 컬러 결정
2.  **Selection (선택)**: 사용자 취향에 따른 텍스처(Glossy/Matte/Satin) 선택
3.  **Rendering (합성)**: 추천된 컬러와 선택된 텍스처를 원본 입술 위에 물리 기반 렌더링(PBR)으로 합성

### 1.2 아키텍처 다이어그램
```mermaid
graph TD
    UserImage[얼굴 입력 이미지] --> Analysis[퍼스널 컬러 분석 모듈]
    Analysis -->|LAB 분석/계절 판정| Diagnosis[타입 확정 & 컬러 추천]
    Diagnosis -->|추천 컬러(Hex)| RenderEngine[립 합성 엔진]
    UserSelect[사용자 텍스처 선택] -->|Glossy/Matte/Satin| RenderEngine
    RenderEngine --> FinalImage[최종 합성 결과]
```

---

## 2. 퍼스널 컬러 진단 엔진 (Diagnosis Engine)

본 시스템은 머신러닝(Black-box)이 아닌, **설명 가능한 규칙 기반(Rule-based)** 알고리즘을 사용합니다.

### 2.1 입력 데이터 및 전처리
-   **입력**: 조명(CCT값) 및 화이트밸런스(WB)가 통제된 얼굴 정면 이미지.
-   **전처리**: MediaPipe FaceMesh를 이용해 피부 ROI(Region of Interest)를 추출하고 평균 LAB 색상 값을 계산합니다.

### 2.2 진단 알고리즘 파이프라인
1.  **파생 지표 계산**:
    -   채도($C$) = $\sqrt{a^2 + b^2}$
    -   웜/쿨 지표($T$) = $b + 0.2a$ (b* 값이 온도 판단의 핵심)
2.  **계절(Season) 판정 (Stage 1)**:
    -   사용자의 $T$ 값과 각 계절 기준값 사이의 거리를 계산하여 가장 가까운 계절을 확정합니다. (웜/쿨 분리)
3.  **세부 타입 확정 (Stage 2)**:
    -   확정된 계절 내 세부 타입들과의 색차($\Delta E$) 및 패널티 점수를 계산합니다.
    -   **Final Score** = $\Delta E + \lambda_L P_L + \lambda_C P_C + \lambda_T P_T$ (명도, 채도, 온도 패널티의 가중합)
    -   최소 점수를 가진 타입을 최종 퍼스널 컬러로 확정합니다.

---

## 3. 립 시뮬레이션 렌더링 엔진 (Rendering Engine)

단순한 2D 색상 합성이 아닌, 입술의 입체감과 광학적 특성을 고려한 **2.5D 물리 기반 렌더링**을 수행합니다.

### 3.1 렌더링 파이프라인
1.  **Lip Segmentation**: 입술 영역 마스크 추출
2.  **LCS (Lip Coordinate System)**: 입술 전용 좌표계 구축
3.  **Fake Normal Generation**: 기하학적 곡률 + 미세 텍스처 → 법선 벡터 생성
4.  **Lighting Calculation**: Specular(반사광) 및 Clear Coat(코팅광) 연산
5.  **Compositing**: Base Color + Lighting Layers 합성

### 3.2 주요 중간 산출물 (Intermediate Maps) 상세 가이드

렌더링 품질을 결정짓는 핵심 Map들의 역할과 생성 원리는 다음과 같습니다.

#### A. Geometry & Coordinate Maps (기하 구조)
| Map 이름 | 설명 및 역할 |
| :--- | :--- |
| **`lip01`** (Soft Mask) | 입술의 전체 영역을 정의하는 0~1 마스크입니다. 경계선에 **Feathering(부드러운 처리)**이 적용되어 있어, 립 컬러가 피부에 자연스럽게 스며들도록 합니다. |
| **`x_hat`** | 입술의 좌측 끝(-1)부터 우측 끝(1)까지의 **가로 정규화 좌표**입니다. 입술의 대칭성을 제어하는 데 사용됩니다. |
| **`y_hat`** | 입술 중앙선(0)을 기준으로 윗입술 끝(-1), 아랫입술 끝(1)을 정의하는 **세로 정규화 좌표**입니다. 윗입술과 아랫입술을 논리적으로 구분하고 깊이감을 부여하는 핵심 기준입니다. |
| **`r`** | 입술 중앙선으로부터의 수직 거리($|y\_hat|$)입니다. 입술 중앙에서 외곽으로 갈수록 값이 커지며, **하이라이트의 감쇄(Falloff)**나 볼륨감을 계산할 때 사용됩니다. |

#### B. Physics Maps (물리 속성)
| Map 이름 | 설명 및 역할 |
| :--- | :--- |
| **`normal_Nx / Ny / Nz`** | 각 픽셀이 바라보는 방향을 나타내는 **법선 벡터(Normal Vector)**입니다.<br>- **Nx**: 좌우 경사<br>- **Ny**: 상하 경사 (입술의 볼륨감 결정)<br>- **Nz**: 정면 돌출 정도<br>기하학적 곡률(Macro)과 이미지의 미세 주름(Micro)을 결합하여 생성됩니다. |
| **`normal_length`** | 생성된 법선 벡터의 크기입니다. 렌더링 오류가 없다면 모든 입술 영역에서 균일한 값(1.0)을 가져야 합니다. |

#### C. Lighting Maps (조명 효과)
| Map 이름 | 설명 및 역할 |
| :--- | :--- |
| **`spec`** (Specular) | **직접 반사광** 영역입니다. 법선 벡터와 조명 방향(`LIGHT_DIR`)의 내적(Dot Product)을 통해 계산됩니다. 립스틱 자체의 유분기나 수분감에 의한 1차 광택을 표현합니다. |
| **`hwf`** (Highlight Weight Function) | **Clear Coat(투명 코팅막)가 맺힐 최적의 위치**를 지정하는 가중치 맵입니다. `LCS` 좌표계를 기반으로 입술 산(Cupid's bow)이나 아랫입술 중앙 등에 하이라이트를 강제로 위치시켜 미적으로 아름다운 결과를 유도합니다. |
| **`clear_spec`** | **2차 코팅 광택**입니다. `hwf` 영역 내에서 계산되며, 립글로스나 탕후루 립처럼 입술 위에 씌워진 투명하고 매끄러운 막의 반짝임을 표현합니다. 미세한 주름을 메우는 효과(Texture Smoothing)가 반영됩니다. |

---

## 4. 환경 설정 및 배포 (Environment)

이 프로젝트는 Python 기반의 Streamlit 애플리케이션으로 구성됩니다.

### 4.1 필수 요구 사항
-   **Python**: 3.10 이상 (권장 3.13)
-   **Core Libraries**:
    -   `Opencv-python` (이미지 처리)
    -   `Numpy` (행렬 연산 - **버전 호환성 주의**)
    -   `Mediapipe` (얼굴 랜드마크 추출)
    -   `Streamlit` (웹 UI)

### 4.2 실행 방법
다른 환경에서 프로젝트를 실행할 경우, 반드시 기술된 의존성 버전을 준수해야 합니다.
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate

# 의존성 설치 (버전 고정 필수)
pip install -r requirements.txt

# 애플리케이션 실행
streamlit run app.py
```

### 4.3 주의 사항
-   `setup_guide_for_ai.md` (또는 `setup_guide.md`) 문서는 AI 에이전트가 이 환경을 자동으로 구성할 때 필요한 프롬프트를 포함하고 있습니다.
-   `requirements.txt`에 명시된 버전은 `modules/` 내의 알고리즘이 정상 동작함을 보장하는 버전입니다. 임의 업데이트 시 호환성 문제가 발생할 수 있습니다 (예: Numpy 2.0 이슈).
