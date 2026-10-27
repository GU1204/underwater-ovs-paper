可以把 `test` 最终 mapping 文档的生成过程概括成下面这套流程。原始输入只是 `category_name + annotation_count`，例如 `Abraliopsis`、`Abyssocucumis abyssorum`、`Acanella` 等，共 963 个原始名称。

1. **先统计原始类别和长尾分布。** 不直接把 963 个名称当成同等级类别，因为其中混合了 species、genus、family、order、普通名称、设备和垃圾标签，而且很多类别只有 1–2 个实例。
2. **放弃把所有类别压到 FathomNetCLEF 的 32 类。** 32 类可以作为 taxonomy 参考，但对开放词汇分割过粗，例如大量形态差异很大的鱼都会变成 `bony fish`。因此最终目标改成“中等粒度、视觉可识别的 concept vocabulary”。
3. **按照“视觉可分性”而不是统一 taxonomy rank 建立视觉概念。** 只有当多个原始标签在形态和语义上足够一致时才合并，例如：
   `Acanella / Isididae → bamboo coral`，
   多个典型海葵类 → `sea anemone`，
   某些深海长臂虾 → `deep-sea caridean shrimp`。
   最终 mapping 中同时保留原始 `category_name` 和新的 `visual_concept`，方便回溯。
4. **参考 MARIS 的类别粒度校正命名。** 不要求所有类都处在同一个 taxonomy rank，而是优先使用对视觉模型有意义、文本编码器能理解的名称，比如 `sea anemone`、`sponge`、`sea fan`、`squid`、`can`、`anchor` 等。最终表中专门保留 `maris_reference` 和 `maris_relation` 两列记录这种对应关系。
5. **禁止 `other fish / other shrimp / other jellyfish` 这类残余类。** 这类名称只表示“剩下的东西”，没有稳定的视觉或语言语义。因此剩余标签必须二选一：映射到一个有明确含义的视觉概念，或者进入 `IGNORE`。
6. **明确使用 `IGNORE`。** 不是所有原始 label 都必须被强行保留。像 `Actinopteri`、`Actinopterygii` 这种范围过宽、不能定义稳定视觉目标的标签直接忽略；极低频且没有足够依据建立视觉类别的 taxon 也可以忽略。最终文件里这些类别对应 `visual_concept=-1/IGNORE`，并给出具体理由。
7. **人工物体不因为“非生物”而删除。** 对 OVS 来说，只要语义明确、视觉上能识别，就可以保留。例如 `can`、`anchor`、`fishing trap`、设备等。垃圾类进一步细化：`can → can`，`plastic → plastic debris`，但含义过于模糊的 `trash → IGNORE`。
8. **划分 CORE / OPTIONAL / IGNORE，但不单纯依赖样本数。** 高频、稳定、任务重要的概念进入 `CORE`；低频但语义明确的概念可以 `OPTIONAL`。同时允许低频例外，例如 `can` 虽然样本少，但类别非常明确，并与 MARIS 类别直接对应，所以最终提升为 `CORE`。同理，`sea slug` 和 `octopus` 也最终设为 CORE。
9. **最后做文本语义增强。** 对视觉概念增加 `suggested_text_query` 和 `synonyms`。例如 `sponge → sea sponge`，`sea fan → fan coral；gorgonian`，`sea anemone → anemone；actinian`。如果没有可靠的常见名或同义词就留空，不为了填满字段而制造名称。最终 mapping 已经把这些信息作为独立字段保存。

最终得到的不是简单的：

963 raw labels→N taxonomy classes963\ \text{raw labels}\rightarrow N\ \text{taxonomy classes}

而是：

raw category→visually recognizable concept→CORE / OPTIONAL / IGNORE\boxed{ \text{raw category} \rightarrow \text{visually recognizable concept} \rightarrow \text{CORE / OPTIONAL / IGNORE} }

并附带：

text query+synonyms+MARIS reference+mapping confidence\text{text query}+\text{synonyms}+\text{MARIS reference}+\text{mapping confidence}

所以最终 `test mapping` 文档本质上是一个**面向开放词汇水下分割的视觉语义本体映射表**，而不是纯 taxonomy mapping。当前最终表中的主要字段已经包括 `category_name`、`annotation_count`、`visual_concept`、`visual_supergroup`、`train_tier`、`mapping_action`、`visual_confidence`、`suggested_text_query`、`synonyms`、`maris_reference` 和 `rationale`。

如果写进论文的方法部分，可以浓缩成一句：

> **We converted the heterogeneous fine-grained FathomNet annotations into a medium-granularity visual vocabulary based on visual separability, semantic clarity, task relevance, and MARIS-informed category definitions, while discarding ambiguous labels and enriching retained concepts with common names and synonyms for open-vocabulary segmentation.**