1. 谨记命名规范：
   - 所有的**类名、文件名**，大驼峰：    GetUserMessage， UserInfoTable
   - 所有的**参数名**，小驼峰： userID， userName
   - 注意缩写的部分，需要都用全大写。 包括： API， ID 是特殊的。 如：EComment， TXService
   - 命名里面避免出现  _  等特殊字符。
   - 项目名，不能重复。
   - 同一个项目里面的 微服务名，不能重复；（不同项目的微服务如果重名的话，toX微服务Message，这个会重复（ 待解决，Todo @刘毓  @李阳 ）
   - **API**，要以Message 结尾； 其Planner 要跟Message名字完全对应上，以Planner结尾；
   - **Table名**，注意一定要大驼峰。
   - 文件夹内，同一层的文件，不要重名，需要注意大小写（不要相同文件名，大小写不一样，不然gitlab仓库可能会出现问题。）
2. 论文主线梳理
   1. 你是论文写作思路提炼助手。按原文顺序，仅概括每连续1–3句话的主旨或内容，不评价、不补充、不改变论述结构。保留原有标题和段落边界；每段使用无序列表，格式为“- [num] 简短概括”，其中[num]表示该条概括的原文句数。概括可用短语或半句话，尽量删除重复主谓、虚词及“为了、旨在、指出”等无实义表达；不得跨段合并。

clip语义特征和structure特征之间存在互补，弱mask生成粘连或错误区域，那对应的clip 语义特征应该也有问题？

需要回答的主要问题：

1. 为什么是频域adapt（参考Visual Fourier Prompt Tuning   这里讲了加入频域特征的原因以及和 F3M），高频和低频分别有什么含义和作用，又是如何设计的。
2. 为什么只在浅层？（参考RevisitingthePowerofPromptforVisualTuning和fcclip的设计，这是一个层级结构，stage3和4 的设计语义太强了，实验效果总是很糟糕）
3. 为什么设计成现有FFT设计？（从Explicit-Visual-Prompt搬来的，但需要讲明白设计or搬这个起到的作用是什么？可以参考和总结EVP的原文）
4. 为什么设计要设计SCORE？（SAN maskadapter PosSAM_2024，但这三个都不是水下叙事，DiveSeg_2026和OVRCOAT_2026都是把他们关于这部分的设计上升到一种更高的作用，然后再讲如何设计的）
5. 另外，现在的叙事太偏向于在fccip上改进，需要强调的主线是水下开放词汇分割，这两个模块或者说“score和mask2former”“频域ADAPT+FROZEN CLIP”到底在水下开放词汇分割中起到了什么作用？可以参考Generalization Boosted Adapter for Open-Vocabulary Segmentation和DiveSeg_2026和OVRCOAT_2026关于这部分的叙事。

对于开放词汇任务，我们真正关心的并不是“特征分布是否与自然图像一致”，而是预训练视觉语言模型内部已有的开放语义知识，能不能在新的视觉条件下被有效调用。

尽可能保留已有开放语义空间，同时调整视觉证据，使其重新能够被这个语义空间有效利用。

segmentation 分支学到的 task-specific representation；

pretrained CLIP 保留的 open-vocabulary representation。

Can the model use the resulting region evidence to invoke the correct semantic concept?

如何从水下观测中形成足够具有判别性的对象证据？

这个区域究竟支持哪个开放词汇概念？

Semantic Accessibility
      ↓
two evidence bottlenecks

1. Visual Evidence Formation
   → CGDI
   → shallow
   → frequency-aware

2. Region Evidence Interpretation
   → SCORE
   → adapted task semantics + pretrained semantics
   → region-level coordination

region semantic evidence aggregation

第一处决定：

> **水下图像被编码成什么样的视觉表示。**

第二处决定：

> **一个预测实例最终以什么样的区域表示进入文本匹配。**

### 第一处不是简单的 backbone interface

它实际上是：

> **underwater observation → task-usable visual representation**

水下外观变化使自然图像预训练得到的视觉表示未必直接适合当前场景，所以你研究“视觉 representation 如何形成”。

于是频域只是回答：

> 我用什么额外视觉信息帮助这个 representation formation？

### 第二处也不是简单的 classifier interface

它是：

> **predicted instance → text-matchable region representation**



1. 图1：如何展示水下退化？或者水下退化影响模型表征，参考：水下论文和之前退化会对clip表征干扰的论文，开题报告里有。
2. 图2：整体论文框架，参考possam，fcclip、diveseg
3. 关于数据集的表述参考，pangOpenVocabularyCamouflagedObject2024，实验设置也参考这个。（这个里面没有很多数据集，表述会比较接近我们的目标）



1个问题，当前motivation是否影响开放词汇分割叙事，如何保证这一点不被抹除，我的工作不止局限于水下视觉特征学习？

可能的解决方案，我只在创新点1引入视觉线索，最小程度上减少破坏clip语义；创新点2尽可能包装到语义方向，另外提出，不针对水体风格对图像做任何增强，而是在text encoder侧调整prompt（MARIS），迁移文本语义到水下。

Underwater degradation weakens visual evidence, while CLIP-based region recognition underutilizes the object-centric evidence already learned by the segmentation branch.

Region Recognition Deficiency





**一个要点** 

mask 是在 clip 特征上学来的，但是 封闭数据集，mask本身是一个 Task-trained region representation

单一来源的 region semantic evidence 难以同时充分利用 pretrained open-vocabulary semantics 与 task-adapted region evidence，因此需要协调两类互补的区域语义证据。

CLIP → semantic anchor / receiver 

task representation → proposal-specific correction source

**你用 task-learned region evidence 去 refine CLIP region semantics，但为什么这种 task evidence 不会反过来加剧 seen bias？** 你的 `correction-only + keep-or-correct + 保留原始 CLIP OOV branch`

Decoupling Zero-Shot Semantic Segmentation认为 mask2former生成了 class-agnostic mask embedding 用于mask loss和 semantic embedding 用于 class loss，推理时用另一个frozen clip 去编码 cropped image来获取开放能力。这和fcclip用 mask pooling没什么太大区别。

possam提到，只用discriminative clip feature做有限集上的监督，会让它产生 seen bias，sam里面时class agnostic feature，所以二者做结合，保证开放词汇能力。

我认为，这里存在谬误，只用clip feature做监督，本质上只在监督mask，因为clip feature是frozen的，没有意义；对我这样在微调的clip做监督，确实存在有限集上的bias，破坏模型的开放词汇分割能力。

另外，只在mask2former生成的semantic embedding上做监督效果也是不好的，因为其中的semantic是有限的 ，只在seen class的，且这个semantic embedding来源的transformer feature 同时服务于mask generation。这是一个只有seen类能用的语义，unseen类完全无法利用这个分支。

所以我们需要增强这个部分。将clip semantic作为语义锚点，让模型学习如何利用mask feature中包含的更偏向于空间、轮廓、结构semantic

![image-20260827031308133](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20260827031308133.png)

domain adaptation–open-vocabulary retention trade-off in underwater scenes





vlm（such as clip）在开放词汇有优势（Pretrained vision-language models (VLMs) provide strong open-vocabulary recognition by transferring broad visual-semantic knowledge from large-scale image-text pretraining. ）

然而在水下实例分割时，面临缺少细粒度特征（或者说，mask级别特征）和数据集受限的问题。

现有方法学习一组dataset level的模型参数（这里表达开放域上的问题被一个dataset 上的监督学习代表了，学到的模型参数向闭集偏移，大概这个意思吧，我不知道怎么表述。其实这里应该不光是微调方案，clip模型本身，和为了下游任务接的mask generator也是在有限数据集上训练的）缺少建模（？是这个词吗？或者说ignore）image 级别图像之间差异的能力（也就是说忽视域间变化），也不利于泛化到新类别和应对新的视觉分布（这个概需要描述和训练数据集不同的风格或者分布 ）

为了缓解上述问题，充分利用（或者说保留）VLM的开放能力，建了一个水下开放词汇分割方法（名字还没想好，大概还要加点限定词）。

首先图像依赖频域适应：从每个input图像生成频域特征，分解为低频和高频，用来引导或者prompt或者remodulate pretrained representation，因为它们缺乏边缘和纹理信息（或者就说丰富局部/空间特征）（这里有很多原因，但我不知道如何总结和取舍：一方面，clip这样的模型是为了整图语义建立的，更倾向于语义特征，缺乏空间建模能力，很多边界和纹理特征被忽略，这首先不利于分割；另外水下图像的退化是一个复杂的过程很难建模，我需要一种视觉分布变化鲁棒的特征来改善这个特征，很自然的想到了用这张图像的的高频细节特征，但这其中存在噪声和伪影，所以我借用低频中的粗糙空间特征来对高频进行gate，但这里不需要讲这么细。另外为了尽可能少破坏原有语义空间，这个模块只放在模型浅层）；

另外mask 依赖语义xx：实现 region recognition是另一个重要步骤，为了防止向特定类别偏移，从clip的层级特征中学到mask generation feature，希望这个mask generator 只学到如何利用clip特征生成 类别无关 或者说 语义无关 的 mask，所以mask loss的feature可以传给它，classification loss不传给它；那么如何实现classification loss呢？用 clip region feature和mask region feature做cross attention来改善clip region feature，期待模型学习如何利用mask region feature来使clip region feature具有region级表征能力，因为相较clip有更多的空间和结构特征。

（后面两个方法部分如何和前面的motivation部分对应我还没想好，该如何取舍和包装请你给出建议。）

