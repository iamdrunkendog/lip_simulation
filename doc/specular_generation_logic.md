# 스펙큘러(Specular) 하이라이트 생성 로직 상세 설명

본 문서는 현재 프로젝트의 코드를 바탕으로 입술 시뮬레이션에서 광택(Specular)이 어떻게 계산되고 렌더링되는지 기술합니다.

---

## 개요: 렌더링 파이프라인
현재 엔진은 물리적 자연스러움과 예술적 제어력을 동시에 확보하기 위해 다음과 같은 단계로 렌더링을 수행합니다.

1. **LCS 생성**: 입술의 해부학적 좌표계 구축
2. **Fake Normal 생성**: LCS 곡률 + 마스크 경사도 + 질감을 결합한 법선 벡터 생성
3. **Lighting 연산**: 조명 방향과 법선 벡터의 물리적 상호작용 계산
4. **Post-Processing**: 선명도 조절 및 최종 블렌딩

---

## 1. 법선 벡터(Normal Map)의 구성
`modules/fake_normal.py`는 두 가지 데이터를 혼합하여 최종 법선 벡터($N$)를 만듭니다.

- **Layer 1 (Mask Gradient)**: 입술 마스크의 경사도를 소벨 필터로 계산하여 입술의 거시적인 두께감을 형성합니다.
- **Layer 2 (Image Texture)**: 사진 속의 실제 명암 차이(High Frequency)를 추출하여 입술의 미세한 주름과 질감을 표현합니다.

이들이 합쳐져 각 픽셀이 "어느 방향을 바라보고 있는지"를 나타내는 RGB 맵(`normal_Nx/Ny/Nz`)이 완성됩니다. 오늘 시도했던 LCS 기반 전역 곡률 주입은 하이라이트가 과하게 중앙으로 쏠리는 현상으로 인해 현재는 제외되었습니다.

---

## 3. 물리적 조명 연산 (Lighting Interaction)
`modules/specular.py`에서 실제 반사광의 위치를 결정합니다.

```python
# modules/specular.py
ndotl = np.clip(Nx*Lx + Ny*Ly + Nz*Lz, 0, 1) # 내적(Dot Product)
spec = (ndotl ** shininess) * strength
```

- **내적(Dot Product)**: 조명 방향($L$)과 표면 방향($N$)이 일치할수록 강한 빛을 냅니다.
- **거듭제곱(Shininess)**: $ndotl$ 값(0~1)에 지수를 적용하여, 빛이 집중되는 영역을 좁히고 날카롭게 만듭니다. 지수가 높을수록(예: 60 → 400) 더 유광(Glossy)인 느낌을 줍니다.

---

## 4. 최종 블렌딩 (Screen Mode)
계산된 하이라이트를 컬러 입술 위에 얹습니다.

```python
# modules/specular.py
def screen(base, add):
    return 1.0 - (1.0 - base) * (1.0 - add)
```
- **Screen Blending**: 하이라이트($add$)가 밝을수록 원본($base$) 색상을 하얗게 덮어씌웁니다. 
- 결과적으로 입술의 원래 색상 질감을 해치지 않으면서도, 조명을 직접 반사하는 **순백색의 반짝임**을 만들어냅니다.

---

## 요약
현재 코드는 **"해부학적 곡률(LCS) 위에 물리 법칙(Dot Product)을 적용"**하는 방식을 사용하고 있습니다. 
단순히 마스크를 씌우는 방식보다 훨씬 입체적이고, 조명 방향에 따라 광택이 자연스럽게 흐르는 결과물을 만들어냅니다.
