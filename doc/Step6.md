# Step 6：CAD/FEA 工具链自动调用（CAD/FEA Tool-Use Automation）

本步骤关注：**让 LLM/Agent 能够自动调用 CAD 和 FEA 工具链，完成从几何建模 → 网格 → 载荷与边界条件设置 → 求解 → 结果解读的整条流水线**。

---

## 1. 步骤定义与成功标准

### 1.1 定义

本步骤不再只是"会算"、也不只是"会建模"，而是：

让 LLM / Agent 能够自动调用 CAD 和 FEA 工具链，完成从几何建模 → 网格 → 载荷与边界条件设置 → 求解 → 结果解读的整条流水线。

**典型形式是**：

- 由 LLM 作为 Planner / Orchestrator；
- 工具侧是：
  - CAD：FreeCAD / CadQuery / 其他 CAD API；
  - FEA：CalculiX、COMSOL、Code_Aster 等；
- LLM 通过工具接口：
  - 生成脚本 / API 调用（例如 Python + FreeCAD + CalculiX）；
  - 执行仿真；
  - 读取结果并迭代改进设计。

### 1.2 成功标准

一个在本步骤表现"合格"的 Agent，至少要做到：

#### 工具调用成功率高

- 生成的 CAD/FEA 代码或 API 调用在环境中可以正常运行，不报语法或逻辑错误；
- 高质量系统（如 CAD-Assistant、FEABench 中的 agent）可达约 80–90% 的可执行 API 调用成功率。

#### 仿真设置物理合理

- 单位、材料、载荷、约束设置合理，不出现严重物理错误（例如固定错误的面、漏掉关键约束等）；
- AutoFEA 专门用 GCN-Transformer 检索 + LLM 生成来减少 FEA 输入中的"幻觉错误"。

#### 结果数值可信

- 与基准 FEA 设置相比，应力 / 位移结果在可接受误差范围内；
- 对明显错误结果（如应力无限大、仿真不收敛）有自动诊断和重试逻辑。

#### 闭环迭代能力

- 能根据 FEA 结果自动提出修改（如增加厚度、加筋板），并重新调用 CAD+FEA 工具验证；
- 典型例子是 FeaGPT 所强调的"GMSA：Geometry–Mesh–Simulation–Analysis 全闭环"。

## 2. 具体案例示例

### 案例：让 Agent 自动检查 L 型支架的安全性并调整厚度

**输入**（自然语言）大致是：

> "已有一个 L 型支架的参数：腿长 100×80 mm，厚度 8 mm，材料为 Q235 钢，端部载荷 300 N。  
> 请用 FreeCAD + CalculiX 做静力分析，计算最大 von Mises 应力和安全系数。如果安全系数 < 2，请自动增加厚度并重算，直到满足要求。"

**理想的 Agent 行为链**：

1. **解析设计参数，调用 CAD 工具**（例如 FreeCAD/CadQuery）生成几何：
   - 生成 L 型支架的 3D 模型；
   - 在模型上标注载荷位置与约束区域（如固定在一面壁上）。

2. **调用 FEA 工具**（如 FreeCAD FEM + CalculiX）：
   - 自动划分网格（默认网格尺寸或自适应）；
   - 设置材料属性（弹性模量、屈服强度）；
   - 设置约束（固定支撑）和载荷（端部面压力或节点力）。

3. **运行求解，读取结果**：
   - 最大 von Mises 应力；
   - 最大位移；
   - 仿真是否收敛。

4. **评估安全系数**：
   - SF = σ_yield / σ_max
   - 若 SF < 2，则：
     - 返回 CAD 模型，增大厚度（例如 8 → 10 → 12 mm），重复步骤 2–4。

5. **输出总结**：
   - "最终选择厚度 12 mm，最大应力 120 MPa，屈服强度 250 MPa，安全系数约 2.08，满足要求。"

这就是典型的 CAD/FEA Tool-Use 闭环。

## 3. 专业模型 / 系统（以"工具使用"为核心）

这里重点列出已经明确实现"LLM + CAD/FEA 工具调用"的系统，而不是单纯的 CAD 生成模型。

### 3.1 CAD-Assistant：面向 CAD 的 Tool-Augmented VLLM（可参考工具化范式）

- **是否开源**：代码与工具定义公开，主要用于研究（非商业许可）。
- **主页**：CAD-Assistant 项目页

**核心特点**：

- 使用 Vision+LLM 作为 Planner；
- 将 FreeCAD Python API + 自定义工具函数封装成一组 Tool（如草图提取、实体布尔运算、渲染、几何分析等）；
- Planner 生成 Python 代码，在带有 FreeCAD 的 Python 解释器中执行，获取几何反馈，再做下一步决策；

**输入 / 输出形态**：

- **输入**：
  - 文本 + 图片（草图、渲染图、3D 扫描）；
- **输出**：
  - FreeCAD Python 代码（例如创建草图、拉伸、布尔运算）；
  - 最终的 CAD 模型（.FCStd）和视觉结果。

**与 Step 6 的关系**：

虽然 CAD-Assistant 目前主要针对 CAD 任务，而非全 FEA 流程，但它提供了极好的"Tool-Augmented 框架样板"，包括：

- 如何把 CAD API 封装成工具；
- 如何在 Planner 中交替生成"自然语言计划"和"代码行动"。

你可以用类似模式把：

- FreeCAD/CadQuery（几何）
- FreeCAD FEM + CalculiX / COMSOL / Code_Aster（FEA）

组合成自己的 CAD/FEA tool-server。

### 3.2 AutoFEA：LLM + GNN 的 FEA 输入生成系统

- **是否开源**：论文公开，系统实现和 512 个 FEA 项目数据集目前主要是论文中描述，未看到完整开源代码。
- **论文**：AutoFEA: Enhancing AI Copilot by Integrating Finite Element Analysis Using Large Language Models with Graph Neural Networks（AAAI 2025）。

**核心思路**：

- **目标**：减少 LLM 为 FEA 生成输入文件时的"幻觉"和错误；
- **管线包括**：
  - Dataset preparation：整理 512 个高质量 FEA 项目作为样本；
  - GCN-Transformer Link Prediction：在这些项目的"工况/步骤图"上做图建模，预测哪些历史项目与当前任务最相似，用于代码检索；
  - LLM code generation：在检索结果的基础上由 LLM 生成 FEA 输入 deck，减少从零编写导致的错误；
- **结果表明**：
  - AutoFEA 能显著提高 FEA 仿真成功率，并减少错误设置带来的失败案例。

**与 Step 6 的关系**：

AutoFEA 重点在"LLM 如何生成可靠的 FEA 输入文件"，并通过图检索减少幻觉，非常接近你想要的"CAD/FEA Tool-Use"中的 FEA 部分；不同于 CAD-Assistant 强调"CAD API Action 序列"，AutoFEA 更偏向"生成完成度很高的 FEA 脚本/输入 deck"。

### 3.3 FeaGPT：End-to-End Agentic-AI for FEA（G-M-S-A 全流程）

- **是否开源**：论文已在 arXiv 公布，代码状态需以项目主页为准（论文摘要中强调完整 GMSA 管线）。
- **论文**：FeaGPT: an End-to-End agentic-AI for Finite Element Analysis（2025）。

**核心特点**：

- 提出完整的 GMSA 流程：
  - Geometry：根据工程需求自动构建或修改几何；
  - Mesh：自动生成适应物理场的网格；
  - Simulation：设置载荷、边界条件、求解参数；
  - Analysis：读取结果并做多目标分析（例如刚度 vs 质量）；
- 强调完全基于自然语言对话接口完成上述流程；
- 使用 CalculiX 作为后端 FEA 求解器，对工业级案例（如 7 叶轮压缩机、12 叶轮涡轮在 110,000 rpm 工况）进行验证，同时对大量 NACA 翼型配置做参数化设计探索。

**与 Step 6 的关系**：

FeaGPT 可以视为目前公开文献中最接近你所设想"Step 6 完整形态"的系统之一：

- 既做几何，也做网格，也做 FEA，最后还做结果分析与设计优化；
- 完全通过 Agent 化 tool-use 对接开源 FEA 工具（CalculiX）。

### 3.4 FEABench：专门评测 LLM 调用 COMSOL 的基准

- **是否开源**：是，代码与任务集合在 GitHub 提供。
- **论文**：FEABench: Evaluating Language Models on Multiphysics Reasoning Ability（Google/哈佛等合作）。

**关键点**：

- **FEABench 关注的是**：
  - LLM/Agent 是否能通过 COMSOL Multiphysics 的 API 正确构造和求解多物理场 FEA 问题。
- **任务设置**：
  - 工程与物理问题以自然语言描述；
  - Agent 需要生成 COMSOL Application Programming Interface 调用（通常是 Java/脚本接口）；
- **自动检查**：
  - API 调用是否可执行；
  - 得到的数值结果是否在正确范围。
- **官方结果**：
  - 他们实现了一个 COMSOL-Agent，能在约 88% 的任务中生成可执行的 API 调用，并对数值结果进行迭代修正。

**与 Step 6 的关系**：

FEABench 是目前最系统性的"LLM FEA 工具调用能力评测集"；虽然使用的是 COMSOL 而非 FreeCAD/CalculiX，但在"Tool-Use 模式"和"成功率指标"上非常具有参考价值。

## 4. 可作为评测环境的专业软件 / 平台

你要做的是：把这些软件包到一个"Tool-Server"里，让 LLM 通过 API / 函数接口调用。

### FreeCAD + CalculiX / Elmer（开源 CAD+FEA）

- FreeCAD FEM 工作台目前主要支持：
  - CalculiX 用于结构和热-力计算；
  - Elmer 用于多物理场（电磁、流体等）。
- 非常适合你在本地或集群上搭建"几何 + FEA"工具服务器。

### COMSOL Multiphysics（商业软件，FEABench 使用）

- COMSOL 提供完整的 API（如 Java、MATLAB 接口），FEABench 就是通过这些接口驱动模拟。
- 若你所在单位有 License，可以直接复用 FEABench 提供的脚本流。

### 其他开源 FEA 求解器

- CalculiX、Code_Aster、Elmer、FEniCS 等都可以集成；
- 对于 Step 6，"有没有良好脚本接口 + 是否容易安装调试"往往比"功能极致强"更重要。

## 5. 如果要训练一个专门的 CAD/FEA Tool-Use Agent：现有可利用的数据与空白

### 5.1 已有"可用但不完全开源"的数据源

#### AutoFEA 的 512 个 FEA 项目

- 用于训练 GCN-Transformer 检索器和 LLM code generation，验证 AutoFEA 工具链；
- 目前论文中只说明"专门构建的数据集"，未看到完整数据开放。

#### FeaGPT 所用工业案例

- 工业涡轮增压器（压缩机/涡轮） + NACA 翼型参数扫描；
- 同样主要在论文中描述，可视为"可复现设置"而非已经开源的数据包。

**结论**：有实战案例，有系统，但数据多数是作者内部构建的，不是直接开源的训练集。

### 5.2 现有可以直接拿来做"工具调用训练/评测"的开源资源

#### FEABench 任务集

- 提供一批工程 / 物理问题 → 标准 COMSOL 工程设置与求解结果；
- 可以用来：
  - 训练 Agent 如何把自然语言转成 COMSOL API 调用；
  - 或至少作为评测集，观察工具调用成功率与数值精度。

#### MechAgents（面向弹性问题的多 Agent FEA 求解）

- 不提供标准训练集，但论文展示了多 Agent 协作解决弹性问题（含生成代码、运行 FEM 等）的模式；
- 可供你设计自己多 Agent 结构时参考。

#### LLM4CAD / CAD-Assistant 等项目的运行日志（可自建）

- 通过在受控环境下运行这些系统，你可以：
  - 收集 "Prompt → Plan → Tool 调用代码 → 工具返回 → 下一步 Plan" 的轨迹；
  - 作为 Tool-Use 模式 finetune 的数据。

### 5.3 必须"自建"的那部分数据

坦白讲，专门面向"CAD/FEA Tool-Use"的开源训练集目前基本是空白区。你真正能做的是：

1. **建一个标准化 Tool-Server**：
   - 提供函数：
     - `build_cad_model(spec)`
     - `run_fea(model, bc, loads)`
     - `extract_results(model_id, metrics)`
   - 尽量保持接口稳定且参数清晰。

2. **自动生成或半自动整理一批任务**：
   - 给定结构、材料、载荷 → 目标是：
     - 求某处应力/位移；
     - 或检查某种设计是否满足约束。

3. **记录 Agent 行为日志，作为 RL 或指令微调的样本**：
   - "正确工具调用轨迹"；
   - "错误调用 + 修正"的过程。

## 6. 如果要评测 CAD/FEA Tool-Use：现有评测集与可行构造方式

### 6.1 直接可用的：FEABench（COMSOL）

**提供**：

- 完整任务描述；
- 参考 API 调用模式；
- 自动打分脚本（是否成功求解、结果是否正确）。

**指标**：

- API 调用可执行率（例如 88%）；
- 正确解答比例（数值误差在容忍范围内的任务比例）。

这可以视为 Step 6 的一个"官方物理工具调用 benchmark"。

### 6.2 你可以自建的：FreeCAD + CalculiX FEA Tool-Use Benchmark

**思路**：

1. **选一组典型 FEA 场景**：
   - 梁、支架、轴、壳体等；
   - 每个场景定义一个标准几何生成脚本（CadQuery / FreeCAD）和目标分析任务。

2. **为每个场景准备**：
   - "标准工具调用序列"：
     - 生成几何；
     - 设置材料、载荷、约束；
     - 调用 CalculiX；
     - 读取结果；
   - 作为 ground truth 轨迹。

3. **评测 Agent 时**：
   - 只给自然语言任务描述（或结构化 JSON）；
   - 让 Agent 从零生成工具调用代码；
   - 检查：
     - 代码是否可执行；
     - 得到的结果是否与 ground truth 接近。

**指标**：

- 工具调用成功率（类似 FEABench）；
- 数值结果精度；
- 自动 debug 能力（第一次失败后能否自修复）。

## 7. 小结：Step 6 在整个链路中的作用与现实情况

### 从链路角度

- 前面的 Step 2–5 更多是在"单一子任务能力"（建模、几何推理、载荷估算）；
- Step 6 是首次把这些能力与实际工程软件工具串成一条可执行流水线；
- 它是从"纸上谈兵"到"真正在 CAD/FEA 里跑起来"的关键门槛。

### 从生态现状看

- **CAD 侧**：已有较成熟的 Tool-Augmented 框架（CAD-Assistant）；
- **FEA 侧**：有 AutoFEA、FeaGPT 等完整或部分自动化系统，证明 LLM + FEA 完全可行；
- **评测方面**：有 FEABench 这样的官方 benchmark，帮你衡量"LLM + FEA 工具调用能力"。

### 但在"训练数据"的层面

- 尚无标准、大规模、完全开源的 CAD/FEA Tool-Use 训练集；
- 真正要做出一个 Step 6 专用 Agent，基本上必须自己搭 Tool-Server + 生成日志数据。

### 现实落地路径

对你来说，这一步的现实落地路径大致是：

1. 参考 CAD-Assistant / FeaGPT / AutoFEA / FEABench 的设计，搭你自己的 FreeCAD+CalculiX 工具服务；
2. 先把 Step 2–5 的能力（建模、几何推理、载荷估算）用"单轮推理 + 单次 FEA 调用"连起来；
3. 再通过日志和 RL/搜索，逐步优化这个 Tool-Use Agent 的鲁棒性和效率。

---

如果你愿意，下一步我可以帮你把整体 1–9 步的 README 结构串起来，或者针对 Step 6 画一版"LLM ↔ ToolServer ↔ FEA"的 ASCII 体系结构图，方便你对接你现在的 MCP / 工具平台。