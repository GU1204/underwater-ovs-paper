# FC-CLIP low_conditioned_gate + highpass 设计源码说明

本文说明 FC-CLIP EVP 中这一组配置的设计和源码路径：

```yaml
MODEL:
  FC_CLIP:
    EVP_ENABLED: true
    EVP_INPUT_TYPE: "fft"
    EVP_PROMPT_TYPE: "highpass"
    EVP_FREQ_RATE: 0.25
    EVP_TUNING_STAGE: 12
    EVP_SCALE_FACTOR: 4
    EVP_HANDCRAFTED_TUNE: true
    EVP_EMBEDDING_TUNE: true
    EVP_ADAPTOR: "adaptor"
    EVP_FUSION_TYPE: "gated_add"
    EVP_GATE_BIAS: -2.0
    EVP_FREQUENCY_ROUTER: "low_conditioned_gate"
```

对应示例配置：

```text
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_evp_high_low_router_stage12_gated_add_gb2_smoke_5k.yaml
```

这条路径的核心思想是：用 FFT high-pass 生成实际注入 CLIP backbone 的 EVP 残差，用同一张图的 FFT low-pass 生成 gate 条件。high-pass 分支负责提供边缘、纹理和局部结构残差；low-pass 分支只决定这些残差应该在什么空间位置被放大或抑制，不直接写回 backbone。

## 源码入口

主要文件：

| 文件 | 作用 |
| --- | --- |
| `fcclip/config.py` | 注册 `EVP_INPUT_TYPE`、`EVP_PROMPT_TYPE`、`EVP_FREQUENCY_ROUTER` 等默认配置。 |
| `fcclip/modeling/backbone/clip.py` | 读取配置，构建 `ConvNeXtEVPPromptGenerator`，并在 ConvNeXt stage/block 循环中调用 EVP。 |
| `fcclip/modeling/backbone/evp_prompter.py` | 实现 FFT prompt、low-frequency conditioned gate、prompt adaptor 和 residual 生成。 |
| `test/test_evp_frequency_router.py` | 覆盖 low-conditioned router 的形状、约束和 residual 等价关系。 |

配置在 `CLIP.__init__()` 中进入 EVP generator：

```text
cfg.MODEL.FC_CLIP.EVP_INPUT_TYPE
cfg.MODEL.FC_CLIP.EVP_PROMPT_TYPE
cfg.MODEL.FC_CLIP.EVP_FREQ_RATE
cfg.MODEL.FC_CLIP.EVP_TUNING_STAGE
cfg.MODEL.FC_CLIP.EVP_FUSION_TYPE
cfg.MODEL.FC_CLIP.EVP_GATE_BIAS
cfg.MODEL.FC_CLIP.EVP_FREQUENCY_ROUTER
```

当前实现只支持 ConvNeXt CLIP backbone；如果在非 ConvNeXt backbone 上设置 `EVP_ENABLED=true`，构造时会报错。

## 总体数据流

以 stage 1/2 为例，整体流程是：

```text
input image
  -> FFT high-pass image
  -> handcrafted prompt pyramid
  -> high-pass stage prompt

input image
  -> FFT low-pass image
  -> handcrafted prompt pyramid
  -> low-pass stage prompt

ConvNeXt stem/stage downsample feature
  -> embedding prompt

for each enabled stage block:
  high-pass stage prompt + embedding prompt
    -> per-block adaptor
    -> high_delta [B,C,H,W]

  low-pass stage prompt
    -> same per-block adaptor
    -> low_condition [B,C,H,W]

  spatial_gate = G(feature, high_delta, low_condition)
  residual = spatial_gate * high_delta
  feature = feature + residual
  feature = ConvNeXt block(feature)
```

关键点：

- high-pass prompt 是实际注入 residual 的 handcrafted 分支。
- embedding prompt 只加到实际注入分支，不加到 low-pass 条件分支。
- low-pass prompt 只进入 gate，不直接加到 feature。
- high-pass prompt 和 low-pass prompt 使用同一套 stage/block adaptor 权重，因此二者被映射到同一个 stage channel 空间后再比较和门控。

## FFT high-pass prompt

源码函数是 `fft_visual_prompt(x, rate, prompt_type)`。

输入要求：

- `prompt_type` 必须是 `"highpass"` 或 `"lowpass"`。
- `rate` 必须满足 `0.0 < rate < 1.0`。
- 输入张量形状是图像 batch `[B, C, H, W]`。

实现步骤：

```text
mask = zeros_like(image)
line = int((H * W * rate) ** 0.5 // 2)
mask[..., H/2-line:H/2+line, W/2-line:W/2+line] = 1

fft = fftshift(fft2(image, norm="forward"))

if prompt_type == "highpass":
    fft = fft * (1 - mask)
else:
    fft = fft * mask

prompt_image = abs(ifft2(ifftshift(fft), norm="forward").real)
```

`EVP_PROMPT_TYPE="highpass"` 时，中心低频区域被置零，保留中心外侧的高频分量。`EVP_FREQ_RATE=0.25` 表示中心 mask 的目标面积约为整幅频谱面积的 25%，实际边长会因为 `int(... // 2)` 有离散取整。

还原后返回的是 `abs(real_ifft)`。这意味着 handcrafted prompt 图不是有符号残差，而是高频响应强度图。

## low-pass 条件分支

`low_conditioned_gate` 不是把 `EVP_PROMPT_TYPE` 改成 low-pass，而是在 high-pass 主分支之外额外生成一条 low-pass 条件分支。

源码逻辑：

```text
handcrafted = _init_handcrafted_from_image(_filter_image(image))

if frequency_router != "none":
    low_frequency_handcrafted =
        _init_handcrafted_from_image(_low_frequency_image(image))
    return [(prompt, low_frequency_prompt), ...]
```

其中：

- `_filter_image(image)` 在 `EVP_INPUT_TYPE="fft"` 时调用 `fft_visual_prompt(image, freq_rate, prompt_type)`，这里的 `prompt_type` 是 `"highpass"`。
- `_low_frequency_image(image)` 固定调用 `fft_visual_prompt(image, freq_rate, "lowpass")`。

所以在当前配置下，`init_handcrafted()` 返回的每个启用 stage 元素都是：

```text
(high_pass_prompt_tokens, low_pass_prompt_tokens)
```

单元测试 `test_low_conditioned_router_uses_low_frequency_gate()` 检查了这一点：启用 `low_conditioned_gate` 后，`handcrafted[0]` 必须是 tuple，且 high/low prompt 形状一致。

## stage prompt 初始化

ConvNeXt stage 内的入口在 `CLIP.extract_features_convnext()`：

```text
handcrafted_prompts = evp_prompt_generator.init_handcrafted(image)
image -> ConvNeXt stem

for stage i:
  if EVP stage enabled:
    x = stage.downsample(x)
    prompt = evp_prompt_generator.init_stage_prompt(
        x, handcrafted_prompts[i], i
    )
```

`init_stage_prompt()` 把 handcrafted tuple 拆成：

```text
handcrafted_feature = high-pass prompt tokens
low_frequency_feature = low-pass prompt tokens
```

同时，如果 `EVP_EMBEDDING_TUNE=true`，它还从当前 stage feature 计算 embedding prompt：

```text
embedding_feature = Linear(stage_feature_tokens)
```

最终返回一个四元组：

```text
(
  handcrafted_feature,      # high-pass tokens
  embedding_feature,        # stage feature tokens
  low_frequency_feature,    # low-pass tokens
  channel_gain              # low_conditioned_gate 下为 None
)
```

对于 `low_conditioned_gate`，`channel_gain` 不使用；它是后续 `low_conditioned_fsp_gate` 和 `low_conditioned_band_amplitude_gate` 的扩展槽位。

## per-block residual 生成

每个启用 stage 的每个 ConvNeXt block 前都会调用：

```text
prompt_residual = evp_prompt_generator.prompt_residual(
    x,
    prompt,
    stage_index,
    block_idx,
)
x = x + prompt_residual
x = block(x)
```

`prompt_residual()` 里有两条分支。

主注入分支：

```text
feat = high_pass_handcrafted_tokens
feat = feat + embedding_tokens
feat = block_adaptor(feat)
high_delta = tokens_to_feature(feat, stage_feature.shape[-2:])
```

low-pass 条件分支：

```text
low_tokens = low_pass_tokens
low_tokens = block_adaptor(low_tokens)
low_condition = tokens_to_feature(low_tokens, stage_feature.shape[-2:])
```

两点需要特别注意：

- low-pass 条件分支没有加 embedding prompt。
- low-pass 条件分支经过的是同一个 `_adapt_prompt_tokens()`，也就是同一个 stage/block adaptor 结构。

默认 `EVP_ADAPTOR="adaptor"` 时，adaptor 是：

```text
per-block Linear(prompt_dim, prompt_dim) + GELU
per-stage shared Linear(prompt_dim, channel)
```

因此同一个 stage 内每个 block 有自己的轻量 MLP，但投影回 channel 的线性层是 stage 共享的。

## LowFrequencyConditionedResidualGate

`EVP_FUSION_TYPE="gated_add"` 时会创建 gate。若 `EVP_FREQUENCY_ROUTER` 是 `low_conditioned_gate`，gate 类选择为 `LowFrequencyConditionedResidualGate`。

输入：

```text
f_clip:             当前 block 前的 ConvNeXt feature, [B,C,H,W]
delta_freq:         high-pass + embedding 得到的 prompt residual, [B,C,H,W]
low_freq_condition: low-pass prompt 得到的条件 feature, [B,C,H,W]
```

形状约束：

- 三者 shape 必须完全一致。
- 支持 `[B,C,H,W]` 和 `[B,N,C]` 两种形式；当前 ConvNeXt EVP 主路径使用 `[B,C,H,W]`。

gate 输入拼接：

```text
delta_norm = ||delta_freq||_2 over channel -> [B,1,H,W]
low_norm   = ||low_freq_condition||_2 over channel -> [B,1,H,W]

gate_input = concat(
  f_clip,
  delta_freq,
  low_freq_condition,
  delta_norm,
  low_norm
)  # [B, 3C+2, H, W]
```

gate 网络：

```text
Conv1x1(3C+2 -> C)
GELU
Depthwise Conv3x3(C -> C)
GELU
Conv1x1(C -> 1)
Sigmoid
```

输出：

```text
gate: [B,1,H,W]
prompt_residual = gate * high_delta
```

`gate` 是空间 gate，不是通道 gate；同一空间位置上所有 channel 共享一个 gate 值。

## 初始化行为

`LowFrequencyConditionedResidualGate.reset_output_layer()` 会把输出层初始化为：

```text
output_proj.weight = 0
output_proj.bias = EVP_GATE_BIAS
```

默认：

```yaml
EVP_GATE_BIAS: -2.0
```

因此训练开始时：

```text
gate = sigmoid(-2.0) ~= 0.119
```

由于输出层权重为 0，初始 gate 与输入内容无关，是一个全局一致的保守注入比例。训练后，输出层权重开始学习，gate 才会依赖 `f_clip`、high-pass residual 和 low-pass condition。

这个初始化设计的目的，是避免训练一开始就大幅改写冻结 CLIP backbone 的特征。

## 与普通 high-pass gated-add 的差异

普通 high-pass gated-add 配置没有 `EVP_FREQUENCY_ROUTER`：

```yaml
EVP_INPUT_TYPE: "fft"
EVP_PROMPT_TYPE: "highpass"
EVP_FUSION_TYPE: "gated_add"
EVP_FREQUENCY_ROUTER: "none"
```

此时 gate 类是 `FrequencyResidualGate`，输入是：

```text
concat(f_clip, high_delta, ||high_delta||)
```

`low_conditioned_gate` 的差异是额外加入：

```text
low_condition
||low_condition||
```

也就是说，high-pass residual 仍是唯一被注入的 residual；新增的是低频条件对空间 gate 的影响。

已有设计文档中记录的 5k smoke 结果显示，`low_conditioned_gate` 的 AP 高于纯 high-pass gated-add，因此后续 `low_conditioned_fsp_gate` 和多频带幅度路由都以它作为基线，而不是替换整条 EVP 主路径。

## 与后续 router 变体的边界

`low_conditioned_gate` 不做通道调制。`prompt[3]` 在这条路径中始终是 `None`。

后续两个 router 是在它之上增加通道增益：

| router | 相对 `low_conditioned_gate` 的额外行为 |
| --- | --- |
| `low_conditioned_fsp_gate` | 从当前 stage feature 的 FFT real/imag 统计预测 `[B,C,1,1]` 通道增益，对 high-pass residual 做通道缩放。 |
| `low_conditioned_band_amplitude_gate` | 从低/中/高频带 log-amplitude 统计预测通道增益，并可用 `EVP_FREQUENCY_MODULATION_STAGE` 限定作用 stage。 |

这两个变体仍保留 low-pass 条件空间 gate。它们不是本文讨论的 baseline 本体。

## text adapter 可见的 low/gate 特征

如果配置了 EVP text adapter 并请求：

```yaml
EVP_TEXT_ADAPTER:
  FEATURES: ["evp_low_res2", "evp_gate_res2", "evp_low_res3", "evp_gate_res3"]
```

`CLIP.extract_features_convnext()` 会在每个 block 调用 `prompt_residual()` 后读取：

```text
evp_prompt_generator.last_low_frequency_feature
evp_prompt_generator.last_gate_map
```

然后按 stage 内 block 求平均：

```text
evp_low_res{s}  = mean(block low_condition)
evp_gate_res{s} = mean(block gate)
```

这些特征在进入 text adapter 前会 detach。它们用于给文本原型提供图像条件风格信号，不会通过 text adapter 反向改变视觉 EVP 分支。

## 配置约束

源码构造阶段会检查：

| 约束 | 原因 |
| --- | --- |
| `EVP_FREQUENCY_ROUTER!="none"` 要求 `EVP_FUSION_TYPE="gated_add"` | router 输出的是 gate 条件，只在 gate 融合中有意义。 |
| `EVP_FREQUENCY_ROUTER!="none"` 要求 `EVP_HANDCRAFTED_TUNE=true` | low-pass 条件来自 handcrafted prompt pyramid。 |
| `EVP_HANDCRAFTED_TUNE` 和 `EVP_EMBEDDING_TUNE` 不能同时为 `false` | 至少要有一个 prompt 来源。 |
| `EVP_TUNING_STAGE` 只能包含 `1,2,3,4` 且不能为空 | 只支持四个 ConvNeXt stage。 |
| `EVP_FREQ_RATE` 必须在 `(0,1)` | FFT mask 需要有效中心区域。 |

如果这些条件不满足，源码会直接抛出 `ValueError` 或 `RuntimeError`，不会静默降级。

## 调试与测试覆盖

运行时可观察字段：

| 字段 | 含义 |
| --- | --- |
| `last_gate_map` | 最近一个 block 的空间 gate，形状 `[B,1,H,W]`。 |
| `last_gate_means[(stage_index, depth_index)]` | 对应 stage/block 的 gate 均值。 |
| `last_low_frequency_feature` | 最近一个 block 的 low-pass 条件 feature，形状 `[B,C,H,W]`。 |
| `last_channel_gain_means` | `low_conditioned_gate` 不使用；后续通道调制 router 使用。 |

单元测试覆盖：

- `test_low_conditioned_router_uses_low_frequency_gate()`：检查 high/low prompt tuple、gate map 形状、low-frequency feature 形状。
- `test_prompt_residual_matches_add_prompt_update()`：检查 `add_prompt()` 等价于 `stage_feature + prompt_residual()`。
- `test_low_conditioned_router_requires_gated_add()`：检查 router 必须搭配 `gated_add`。
- `test_low_conditioned_router_requires_handcrafted_prompt()`：检查 router 必须启用 handcrafted prompt。
- `test_default_router_keeps_two_source_prompt_behavior()`：检查未启用 router 时不会产生 low-frequency 条件槽位。

## 读源码时的最短路径

按下面顺序读最省时间：

1. `fcclip/modeling/backbone/clip.py` 中 `CLIP.__init__()` 的 EVP generator 构造。
2. `fcclip/modeling/backbone/clip.py` 中 `extract_features_convnext()` 的 stage/block 循环。
3. `fcclip/modeling/backbone/evp_prompter.py` 中 `fft_visual_prompt()`。
4. `ConvNeXtEVPPromptGenerator.init_handcrafted()` 和 `init_stage_prompt()`。
5. `ConvNeXtEVPPromptGenerator.prompt_residual()`。
6. `LowFrequencyConditionedResidualGate`。

这条链路读完后，就能完整解释 `highpass` 和 `low_conditioned_gate` 在当前 FC-CLIP EVP 中的实际行为。
