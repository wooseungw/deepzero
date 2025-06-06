파일에 직접 패치하는 과정에서 오류가 발생했습니다.
VS Code에서 readme.MD 를 여신 뒤, 아래 마크다운을 통째로 덮어-붙여 주세요. (기존 내용은 지워도 됩니다.)

⸻


# 📑 커스텀 모델 YAML 가이드

> **DeepZero Model Builder** (`deepzero/core/model_builder.py`)에서 사용할 **YAML-기반 네트워크 정의 문법**을 설명합니다.
>
> ```python
> from deepzero.core import model_builder
> net = model_builder.build_model_from_yaml("configs/my_model.yaml")
> ```
>
> 위처럼 호출하면 곧바로 PyTorch `nn.Module` 객체가 생성됩니다.

---

## 1. 전체 YAML 골격

```yaml
in_channels: 3          # (필수) 입력 채널 수
num_classes: 100        # (선택) 전역 변수, 문자열 치환용
depth_multiple: 1.0     # (선택) 레이어 반복 스케일
width_multiple: 1.0     # (선택) 채널  스케일

layers:
  #   from  repeat  Block                      positional-args…               kwargs
  - [  -1,     1,   ConvBlock,                 64, 3, 2,                      {act: relu, drop: 0.1} ]
  - [  -1,     2,   TransformerEncoderBlock,   128, 4, 8 ]
  - [  -1,     1,   GlobalAvgPool ]
  - [  -1,     1,   LinearBlock,               num_classes,                  {norm: ln} ]
```
## 필드 설명
### from	
입력으로 삼을 이전 레이어 인덱스 (-1 = 직전, -2 = 두 단계 전)
### repeat	
블록 반복 횟수 (depth_multiple가 곱해진 뒤 반올림)
### Block	
``` ConvBlock, LinearBlock, TransformerEncoderBlock, GlobalAvgPool 또는 직접 구현한 블록 ```

positional args	블록별 필수 인자  
• ConvBlock → out_channels, kernel, stride  • LinearBlock → out_features  • TransformerEncoderBlock → dim, heads, mlp_ratio
kwargs	선택 : act, norm, drop 등 공통 하이퍼파라미터


⸻

2. 공통 하이퍼파라미터

옵션	값	적용 블록	기본값
act	silu, relu, gelu, tanh, identity	Conv / Linear	silu
norm	bn, ln, gn, none(null)	Conv(2D) / Linear(1D)	bn
drop	0 ~ 1 (확률)	Conv → Dropout2d, Linear → Dropout	0

{act: gelu, norm: gn, drop: 0.25}


⸻

3. 지원 블록 & 시그니처

블록	생성자 서명	설명
ConvBlock	c_in, c_out, k, s, *, act, norm, drop	Same-padding 2D Conv
LinearBlock	_dummy, out_features, *, act, norm, drop	Flatten → Linear
TransformerEncoderBlock	dim, heads, mlp_ratio	ViT-style 인코더 (출력 4-D 복귀)
GlobalAvgPool	–	(B,C,H,W) → (B,C)

새 블록을 추가하려면

class MyBlock(BlockBase):
    ...

처럼 정의하면 레지스트리에 자동 등록됩니다.

⸻

4. 전역 변수 치환

num_classes: 5
layers:
  - [ -1, 1, LinearBlock, num_classes ]   # ⇒ out_features = 5


⸻

5. 예시 – Tiny ViT Classifier

# tiny_vit.yaml
in_channels: 3
num_classes: 10
depth_multiple: 1.0
width_multiple: 0.5   # 채널 절반

layers:
- [ -1, 1, ConvBlock, 64, 7, 2 ]                         # 112×112
- [ -1, 1, ConvBlock, 128, 3, 2,   {drop: 0.1} ]
- [ -1, 2, TransformerEncoderBlock, 128, 4, 4 ]
- [ -1, 1, ConvBlock, 256, 3, 2 ]
- [ -1, 1, GlobalAvgPool ]
- [ -1, 1, LinearBlock, 512,          {act: relu} ]
- [ -1, 1, LinearBlock, num_classes,  {act: identity, norm: none} ]


⸻

6. 고급 팁

목적	방법
깊이·폭 스케일	depth_multiple, width_multiple 값 조정
Dropout OFF	drop: 0 또는 생략
BatchNorm 없는 Conv	{norm: none}
LayerNorm Conv	{norm: ln} (GroupNorm 1 그룹 활용)
반복 블록별 다른 Act	반복을 나눠 별도 레이어로 선언


⸻

7. 자주 나는 오류 & 해결

오류	원인 / 해결
expected 4D input (got 3D)	Transformer 뒤 Conv 배치에서 3-D 전달 → 레이어 순서·채널 확인
BlockBase.registry KeyError	YAML Block 이름이 실제 클래스와 다름
L이 정사각 격자에 대응되지 않습니다	Transformer 토큰 수가 제곱수가 아님 → 입력 해상도·stride 점검


⸻

8. 빌드 & 테스트 스니펫

import torch
from deepzero.core import model_builder

net = model_builder.build_model_from_yaml("configs/tiny_vit.yaml")
x   = torch.randn(1, 3, 224, 224)

print(net(x).shape)                      # torch.Size([1, 10])
print("params:", sum(p.numel() for p in net.parameters()) / 1e6, "M")


⸻

이 가이드를 참고하여 자유롭게 YAML을 편집하며 나만의 모델을 설계해 보세요!
