# Step 1：机械设计需求理解与约束抽取（Requirements & Constraints Understanding）

本步骤关注：**从人类自然语言需求中，结构化抽取可用于机械设计的目标、功能、约束与参数**，为后续概念设计与参数化建模提供输入。

---

## 1. 步骤定义与成功标准

### 1.1 定义

给定输入（通常为自然语言文本，也可能附带草图、图片）：

- 产品/机械系统的功能需求（做什么）
- 性能指标（承载多少、转速范围、寿命、效率等）
- 约束条件（空间、成本、材料、制造工艺、安全标准等）
- 接口与边界条件（安装界面、连接方式、动力/信号接口）
- 操作/使用场景（工况、环境）

任务是将其转化为**结构化的设计需求表示**，例如：

- 统一 schema 的 JSON（`objective`, `loads`, `boundary_conditions`, `constraints`…）
- 后续可直接喂给 CAD/CAE/优化工具的参数配置

### 1.2 成功标准

一个 agent 在 Step 1 被认为“成功”的典型标准：

1. **抽取完整性（Coverage）**
   - 主要功能、关键性能指标、重要几何/材料/工艺约束基本都被抽到结构化结果中；
   - 漏掉的是次要信息，而非硬约束。

2. **抽取正确性（Correctness）**
   - 数值、单位、方向、条件关系没有严重错误；
   - 不“幻觉”不存在的要求。

3. **结构化质量（Structure Quality）**
   - 输出满足预定义 schema；
   - 适合作为下游 CAD/CAE/优化脚本的输入，而无需大量人工清洗。

4. **可追溯性（Traceability）**
   - 每个结构化条目都能回溯到原始文本的片段；
   - 方便工程师审阅与修订。

---

## 2. 具体案例示例

### 2.1 原始自然语言需求（示例）

> 为一台自动化装配线设计一套输送机构，用于输送 10 kg 的零件箱。  
> 输送速度约 0.5 m/s，线体总长度 8 m，中间需要 2 处 90° 转弯。  
> 整机布局高度不能超过 1.2 m，总宽度不超过 1 m。  
> 工作环境为室内，温度 5–40 ℃，要求噪音尽量低，设备成本控制在 8 万人民币以内。  
> 与现有设备的接口为 500 mm × 500 mm 的平面，中心线上需要保留安装孔。

### 2.2 期望的结构化输出（示意）

```json
{
  "objective": "design conveyor system for automated assembly line",
  "payload": {
    "mass_per_box_kg": 10,
    "convey_speed_mps": 0.5,
    "line_length_m": 8.0,
    "turns": [
      {"type": "elbow", "angle_deg": 90, "count": 2}
    ]
  },
  "layout_constraints": {
    "max_height_m": 1.2,
    "max_width_m": 1.0
  },
  "environment": {
    "location": "indoor",
    "temperature_range_C": [5, 40],
    "noise_requirement": "as low as reasonably achievable"
  },
  "cost_constraints": {
    "currency": "CNY",
    "max_budget": 80000
  },
  "interfaces": [
    {
      "type": "planar",
      "size_mm": [500, 500],
      "features": ["mounting holes on centerline"]
    }
  ]
}
```

一个"合格"的 Step 1 agent，需要从原始文本中自动构造出类似的 JSON。

## 3. 现有模型 / 软件资源（可用于本步骤）

本节列出：可以辅助需求理解与约束抽取的模型和工具，以及它们的功能、输入输出形式，并标注是否开源。

### 3.1 通用大模型（用于需求解析 / 信息抽取）

这些模型本身不是"机械设计专用"，但在信息抽取 / schema 填充 / tool-use 方面很强，可作为 Step 1 的基线或上层 orchestrator。

#### 3.1.1 GPT-4 系列（OpenAI）

- **是否开源**：否（闭源 API 模型）
- **典型代表**：GPT-4o / GPT-4.1 等
- **链接**：OpenAI 官方文档（需在 OpenAI 官网查看）

**主要功能**：

- 高质量自然语言理解、信息抽取、复杂 schema 填充；
- 支持通过函数调用 / tool calling 将结构化需求传给后端 CAD/CAE agent。

**输入**：

- 纯文本或多模态（文本 + 图片，如草图、样机照片）；
- 提示中给出 JSON schema 或 slot 描述。

**输出**：

- 解析后的 JSON / YAML / 代码片段；
- 或对人类友好的需求总结 + 结构化字段。

#### 3.1.2 Claude 3.x 系列（Anthropic）

- **是否开源**：否（闭源 API 模型）
- **链接**：Anthropic 官网

**主要功能**：

- 与 GPT 类似的高质量文本理解与总结；
- 对长文档、多轮对话的"指令遵循"和一致性较好，非常适合解析长规格书。

#### 3.1.3 DeepSeek-V3 / DeepSeek-R1（DeepSeek）

- **是否开源**：是，开源权重（open-weight），MIT License
- **资源**：DeepSeek-V3、DeepSeek-R1 在 Hugging Face 提供模型与权重下载，并采用 MIT 许可证发布，用于推理和再训练。

**主要功能**：

- 通用中文/英文大模型，推理与工具调用能力强；
- 可本地部署，用于企业内网或自建"机械设计助手"。

**输入 / 输出**：

- 与 GPT 类似，可在提示中定义 JSON schema，让模型完成需求解析；
- 也可生成后续 CAD 脚本（如 CadQuery / OpenSCAD）供后续步骤调用。

#### 3.1.4 Kimi-K2（MoonshotAI）

- **是否开源**：开源权重（open-weight），修改版 MIT 许可证
- **资源**：Kimi-K2 在 Hugging Face 以 kimi-k2-32b 等仓库形式提供权重，采用 Modified MIT License；聊天服务本身为商用产品。

**主要功能**：

- 针对 tool-use / agentic 场景进行了强化（官方论文强调合成工具规范、分步规划等）；
- 适合用来作为"需求解析 + 工具编排"的上层大模型，再调用专业 CAD/CAE 工具。

**小结**：上述通用大模型可以作为 Step 1 的"前端大脑"，负责阅读需求文档、问答澄清，并输出结构化 JSON，再交给下游几何/仿真 agent。

### 3.2 机械设计专向 LLM / 框架（与 Step 1 有一定重叠）

尽管这些工作的主要目标往往是 "Text-to-CAD"，但其中的数据与任务设定，可以反向用来训练/评测"从语言中抽象出几何与参数"的能力。

#### 3.2.1 LLM4CAD（UT Austin SiDi Lab）

- **是否开源**：
  - 论文公开；
  - 数据集 "Multimodal Dataset for CAD Model Generation" 通过 Texas Data Repository 提供下载（需要接受相应协议），属于研究用途开放。

- **资源**：
  - **论文**："Large Language Models for Computer-Aided Design (LLM4CAD) Fine-Tuned: Dataset and Experiments"，ASME Journal of Mechanical Design, 2025。
  - **数据集 DOI**："Multimodal Dataset for Computer-Aided Design (CAD) Model Generation"，Texas Data Repository，DOI: 10.18738/T8/KV7HON。

**主要功能**：

- 将多模态大模型用于 CAD 模型生成与理解；
- 包含零部件类别 + 几何 + 图像/草图等多模态信息，可用于从语言/图像推断模型特征。

**输入**：

- 典型任务是从文本或图像生成 CAD 特征/序列（但数据中也存在文本描述 + 对应模型的配对）。

**输出**：

- 模型拓扑、类别、一些几何与材质信息；
- 可扩展成结构化"设计需求"抽取任务。

**数据示例**（示意，非真实条目）：

```json
{
  "part_id": "gear_00123",
  "modality": {
    "image": "images/gear_00123.png",
    "cad_model": "cad/gear_00123.step",
    "text_description": "A spur gear with 20 teeth and a central bore."
  }
}
```

在 Step 1 中，可以把 text_description 视为"简化需求"，训练/评测模型从文本中抽取出牙数、模数、孔径等结构化字段。

#### 3.2.2 CAD-Coder / GenCAD-Code（MIT–IBM, 2025）

- **是否开源**：是（代码与数据集仓库在 GitHub 上，Apache-2.0 License）

- **资源**：
  - **论文**："CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation"
  - **代码仓库**：GitHub: https://github.com/anniedoris/CAD-Coder
  - **数据集仓库**：GitHub: https://github.com/anniedoris/GenCAD-Code（包含 HF 数据链接）

**主要功能**：

- 从 CAD 图像生成 CadQuery Python 脚本；
- 数据集 GenCAD-Code 提供约 16.3 万对 CAD 图像 – CadQuery 代码配对，用于训练/评测。

**输入**：

- 单张 CAD 渲染图像 +（可选）文本提示；

**输出**：

- 一段 CadQuery 代码（Python 函数式脚本），执行后生成 3D 模型。

**数据示例**（根据官方 README 抽象出的形式）：

```json
{
  "image_path": "gencad_im/part_000123.png",
  "cadquery_script": "result = cq.Workplane('XY').circle(10).extrude(5)"
}
```

在 Step 1 场景下，可以对 cadquery_script 做反向解析，构造"几何需求 JSON"，并用它作为 supervision，训练模型从自然语言中预测类似结构化结果。

#### 3.2.3 ABC Dataset（ABC: A Big CAD Model Dataset for Geometric Deep Learning）

- **是否开源**：是，代码 MIT License；数据需遵守 Onshape 使用条款

- **资源**：
  - **官网**：https://deep-geometry.github.io/abc-dataset/
  - **论文**：CVPR 2019, "ABC: A Big CAD Model Dataset for Geometric Deep Learning"

**主要功能**：

- 提供约 100 万个 CAD 模型，每个模型包含参数化曲线与曲面；
- 用于表面法向估计、特征检测、重建等几何学习任务。

**文件格式示例**（来自官方说明）：

- meta.yml：原始 Onshape 文档的元信息（作者、时间等）；
- model.step：STEP 格式 CAD 边界表示；
- model.para.zip：Parasolid 原始参数化边界；
- model.stl2：合并后的三角网格。

**数据示例**（概念性说明）：

```yaml
meta:
  author: "User123"
  created_at: "2018-05-01"
  tags: ["bracket", "mechanical"]
files:
  step: "chunk_0001/model_000001.step"
  stl: "chunk_0001/model_000001.stl2"
```

ABC 更偏几何任务，但可作为后续步骤（拓扑优化、几何辨识）的基础数据，对 Step 1 而言，可以用来验证从需求解析出的几何参数是否生成合理形状（需要自建桥接）。

### 3.3 CAD 脚本建模工具（可作为 Step 1 之后的"落地环境"）

#### 3.3.1 CadQuery

- **是否开源**：是（基于 Python 的开源库，通常使用 GPL 兼容协议）

**主要功能**：

- Python 脚本驱动的参数化 3D CAD 建模；
- 尤其适合从结构化参数 / JSON 生成模型，方便与 LLM 集成。

**输入**：

- Python 代码（通常通过 cq.Workplane 链式调用）；
- 各种设计参数（厚度、半径、孔径等）。

**输出**：

- 内存中的 3D 形体（可导出为 STEP/STL 等文件）。

**简单示例**：

```python
import cadquery as cq

width, length, height = 10, 20, 5
result = (cq.Workplane("XY")
          .box(length, width, height)
          .faces(">Z").workplane()
          .hole(5))
```

Step 1 中，CadQuery 常被用作验证需求解析结果是否能生成几何可行模型。

#### 3.3.2 FreeCAD

- **是否开源**：是（LGPL-2.0-or-later）

**主要功能**：

- 通用 3D CAD 系统，支持参数化建模；
- 提供 Python 接口，可脚本化操作；
- 是很多研究（例如 Anvil）中构建 CAD–CFD/优化一体化平台的基础组件。

**输入**：

- GUI 操作或 Python 脚本（加载参数、约束、零件关系）；

**输出**：

- FreeCAD 文档（.FCStd）、STEP/STL 文件、尺寸/装配信息等。

## 4. 可用于评测 Step 1 的环境与方法

### 4.1 文本/JSON 级评测环境

#### 标准答案对比（Schema-based Evaluation）

- 构造一批"机械需求描述 + 标准 JSON"的样例；
- 模型输出 JSON 后，计算：
  - 字段级 F1（是否提取到某字段）；
  - 数值误差（例如尺寸、载荷的相对误差）；
  - 单位归一化正确率。

#### 人工工程师打分

- 由机械工程师对解析结果打分（完整性 / 正确性 / 可用性）；
- 可参考 LLM4CAD、CAD-Coder 中类似的人类评估流程。

#### 跨模型一致性评测

- 用 GPT-4 / Claude / DeepSeek-R1 / Kimi-K2 等多个大模型分别解析同一需求；
- 统计它们在关键字段上的一致性，作为"稳定性"指标。

### 4.2 文本 + CAD 的几何验证环境

结合 Step 2 及之后：

**需求 → 参数 → CadQuery 代码 → 几何验证**

- Agent 在 Step 1 输出结构化参数；
- 通过 CadQuery/FreeCAD 自动生成简化模型；
- 检查基本几何约束（如长度、包络尺寸是否满足）。

#### 与公开数据集联动

- 利用 LLM4CAD 或 GenCAD-Code 中已有的文本 / 代码 / 几何三元组：
  - 把文本视为"简化需求"；
  - 将从文本解析出的参数与实际 CAD 代码/几何进行对比。

**典型评测环境组合示例**：

- 需求文本（自建） + 标准 JSON（人工标注）
- CadQuery / FreeCAD 脚本（从 JSON 生成）
- ABC / LLM4CAD / GenCAD-Code 提供的真实几何，用于对比"需求 → 几何"的合理性。

## 5. 训练 Step 1 Agent 的开源数据集

当前并没有一个"专门为机械需求解析设计"的标准开源数据集，但可以组合利用下列资源，通过一定的"弱监督 / 合成数据"方式构建训练集。

### 5.1 Multimodal Dataset for CAD Model Generation（LLM4CAD 数据集）

- **状态**：研究用途开放（Texas Data Repository，需同意条款）

**用途**：

- 把数据集中关于零件的文本描述视作"简化需求"，几何/类别信息视作"标签"；
- 训练模型从短文本中抽取出：部件类别、关键尺寸区间、功能描述等。

**合成训练思路**：

- 用大模型（GPT-4 / DeepSeek-R1）根据已有 CAD 元数据自动扩写长需求描述；
- 构建 "长需求文本 → 短摘要 + JSON 参数" 的训练样本。

### 5.2 GenCAD-Code（面向 CadQuery 的多模态数据集）

- **状态**：开源（GitHub + Hugging Face）

**内容**：

- 约 163k CAD 图像 – CadQuery Python 脚本配对；
- 另有 400 张 3D 打印实物照片数据集 real_photo_test。

**用途**：

- 通过静态分析 CadQuery 代码，提取几何和拓扑参数；
- 生成"伪需求描述"，例如："圆柱体，直径约 20 mm，高度约 10 mm，中间有通孔"等；
- 训练模型从"伪需求"中抽取参数，并与从脚本解析出的参数对齐。

### 5.3 ABC Dataset

- **状态**：开源（MIT 许可证；几何数据受 Onshape 条款约束）

**用途**：

- 提供大量工业零件几何形状，可作为"真实几何分布"的约束；
- 用于训练"几何可行性判别器"，辅助 Step 1 判断某些需求组合是否物理上可行。

### 5.4 其他潜在资源

- CAD-Coder（Guan et al.）中提到的 110k 文本–CadQuery–3D 三元组（目前论文中描述，数据是否完全公开需持续关注）。
- 工程设计论文、标准（ISO/GB）、开源产品说明书等，可通过信息抽取方式构造"真实需求语料库"。

## 6. Step 1 的开源评测集现状与建议

### 6.1 现有开源评测集

截至 2025 年 11 月，没有发现一个专门针对"机械设计需求解析/约束抽取"的标准开源 benchmark。

现有数据集（LLM4CAD、GenCAD-Code、ABC）都更偏向几何建模 / Text-to-CAD / 几何学习，需要二次加工才能直接用于 Step 1 评测。

### 6.2 建议的评测集构建方法

#### 专家标注型评测集

- 选取 50–200 个真实项目的需求文档（脱敏后）；
- 由机械设计专家定义统一 schema，并完成人工标注；
- 对各种 LLM/agent 在该集上的抽取结果进行对比。

#### 合成 + 人工校验

- 用 LLM 从开源 CAD 数据（LLM4CAD、GenCAD-Code）自动生成多样化的"需求描述"；
- 再由工程师抽查、修订一部分样本，形成半合成评测集。

#### 多模型互评

- 将 GPT-4 / Claude / DeepSeek-R1 / Kimi-K2 的解析结果作为"候选集合"，人工从中选取最佳/最差；
- 用于分析各模型在某些需求类型上的优势与弱点。

## 7. 小结：Step 1 的资源与路径

### 通用 LLM（GPT-4/Claude/DeepSeek/Kimi-K2）

作为"需求理解 + tool-use orchestrator"，负责复杂语言解析与 JSON schema 填充。

### CAD 专向模型与数据集（LLM4CAD、CAD-Coder、GenCAD-Code、ABC）

虽然主要面向 Text-to-CAD / 几何任务，但其多模态三元组和脚本化 CAD 表示，提供了构建 Step 1 训练/评测集的坚实基础。

### 几何/仿真环境（CadQuery、FreeCAD）

为 Step 1 抽取的结构化需求提供"落地验证"——用解析出的参数生成几何，并检查是否满足基本物理与几何约束。

在后续 Step 2+ 中，可以继续在此 README 基础上，扩展"概念方案生成""参数化建模""仿真优化"等环节的资源与评测方法。