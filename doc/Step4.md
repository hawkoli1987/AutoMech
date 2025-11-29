# Step 4 – 装配 / 运动学（Assembly & Kinematics）

---

## 1. 步骤定义与成功标准

### 1.1 定义

本步骤关注两类能力：

1. **装配建模（Assembly Modeling）**
   - 将多个零件（包含几何形状和局部坐标系）组合成一个装配体；
   - 为零件之间建立 **约束 / 接口**：
     - CAD 里的 Mate / Joint（同轴、重合、平行、距离、角度等）；
     - 装配图中的装配关系图（Assembly Graph）。
   - 形成层级结构（装配 → 子装配 → 零件），通常对应后续 BOM。

2. **运动学建模（Kinematics）**
   - 在装配约束的基础上，识别每个关节的类型与自由度（DOF）：
     - 旋转副（revolute）、移动副（prismatic）、球副（ball）、固定副（fixed）等；
   - 对给定驱动输入进行 **正运动学**（forward kinematics）；
   - 判断装配是否约束合理（既不过约束，也不欠约束）。

### 1.2 成功标准

一个在本步骤表现“合格”的 LLM / Agent，应当能够：

1. **自动推断装配关系**
   - 给定多零件 CAD 模型，能根据零件几何 & 接触面，提出合理的装配关系：
     - 哪两个零件采用何种 Joint；
     - Joint 的轴线 / 平面如何确定（例如通过孔轴线、同轴圆柱等）；
   - JoinABLe 就是一个针对 parametric CAD joint 学习的代表。

2. **推断运动 DOF / Joint 类型**
   - 对给定装配（已知或预测 joints），输出每个部件的运动形式和 DOF；
   - Mates2Motion 就是通过深度网络从 CAD 装配 + mates 推断真实运动自由度的工作。

3. **在 CAD/仿真环境中执行 & 验证**
   - 在 FreeCAD / 物理仿真引擎（SAPIEN / ManiSkill / 其它）中自动：
     - 建立或修改装配约束；
     - 驱动关节做有限角度/位移的测试；
     - 检查是否出现约束冲突、干涉或机构卡死。

---

## 2. 具体案例示例

**案例：两级减速器中的轴承座 + 轴 + 齿轮装配**

- 输入：
  - 一套零件：箱体、两根轴、4 个滚动轴承、2 个齿轮、若干垫片 / 端盖；
  - 每个零件有完整的 B-Rep 几何与局部坐标系。
- 任务目标：
  1. 识别轴承与箱体之间为 **固定 + 过盈 / 间隙装配**；
  2. 识别轴承与轴之间为 **同轴 + 过盈装配**，并设置旋转自由度在轴相对于箱体；
  3. 识别齿轮与轴之间为 **键连接 / 过盈连接**，视为与轴刚性连接，无相对运动；
  4. 推断整个机构的 DOF：每根轴 1 个旋转自由度。

**Agent 的理想行为：**

1. 读取 CAD 装配数据 / 或分散零件：
   - 如通过 Fusion 360 Gallery Assembly Dataset 的 assembly graph + joints 信息作为训练 / 参考格式。
2. 对零件对之间的接触面进行分析：
   - 圆柱 + 圆柱 → 候选 **revolute joint**；
   - 平面 + 平面大面积贴合 → 候选 **fixed joint**。
3. 输出结构化装配关系与关节：
   - `joint(id="J1", type="revolute", partA="shaft_1", partB="housing", axis=(0,0,1), origin=...)`；
   - `joint(id="J2", type="fixed", partA="gear_1", partB="shaft_1", ...)`。
4. 在 FreeCAD / 物理引擎中执行一个简单运动：
   - 驱动 `J1` 旋转 30°，判断几何干涉是否出现；
   - 若无干涉且转动顺畅，则该装配 & 运动学建模通过基本验证。

---

## 3. 专业模型 / 软件（优先开源）

### 3.1 FreeCAD + 装配工作台（Assembly Workbench）

- **是否开源**：是（LGPL 许可，完整开源）  
- **链接**：
  - FreeCAD 官网：https://www.freecad.org  
  - A2plus 装配工作台（第三方扩展，约束装配）：https://github.com/kbwbe/A2plus   
  - FreeCAD 1.0 引入了官方 Assembly workbench（集成装配能力）  

**主要功能**

- 零件导入与装配：
  - 支持从单独文件导入多个零件（A2plus 等工作台专门面向零件装配）；
- 装配约束：
  - 同轴、重合、距离、角度等装配约束；
  - 通过约束求解器求解零件相对姿态；
- 简单运动测试：
  - 在部分装配工作台 / 教程中，演示了基于约束的“运动仿真”和干涉测试。

**典型输入 / 输出**

- 输入：
  - 若干 `Part` 或 `Body`（从 `.FCStd` 或 STEP/IGES 导入）；
  - 用户或脚本指定的约束（如“此面与彼面重合”、“此圆柱同轴”等）。
- 输出：
  - 解算后的零件位姿（装配结果）；
  - 可通过脚本导出装配树、约束图、关节信息，用作 Agent 的训练反馈或评测环境。

> 对于 Agent，你可以把 FreeCAD 装配操作封装成一组 Tool（如 `add_constraint`, `solve_assembly`, `simulate_joint`），类似 CAD-Assistant 的工具化方式。

---

### 3.2 Fusion 360 Gallery Assembly Dataset + JoinABLe / Mates2Motion

> 这里更偏“专业研究模型 + 数据集 + 训练代码”，但对你要做的 agentic 装配 / 运动学非常关键。

#### 3.2.1 Fusion 360 Gallery Assembly Dataset

- **是否开源**：数据和代码公开下载，用于非商业研究（许可证允许非商业研究使用）。  
- **链接**：
  - 数据集主页：https://github.com/AutodeskAILab/Fusion360GalleryDataset   

**数据内容**

- Assembly Dataset：
  - 来自 Autodesk Fusion 360 Online Gallery 的真实多零件机械装配；
  - 包含 **Assembly Data**（8251 个装配、15 万+ 零件）和 **Joint Data**（32148 个关节）；
  - 关节信息包括 joint 类型、连接零件 ID、接触面、孔等几何信息。
- Reconstruction / Segmentation Dataset（用于建模阶段，可与 Link 2/3 复用）。

**典型样例**

- 一条 Joint Data 记录包含：
  - 两个零件的 ID；
  - 关节类型（如 `revolute`、`slider`、`rigid`）；
  - 关节坐标系、轴向、限制范围等（以 JSON 形式提供）。

#### 3.2.2 JoinABLe（Learning Bottom-Up Assembly of Parametric CAD Joints）

- **是否开源**：代码公开，用于研究（非商业许可证）  
- **代码**：https://github.com/AutodeskAILab/JoinABLe   

**主要功能**

- 从 parametric CAD 装配（Fusion 360）中学习如何预测关节（joint）的类型及参数；
- 基于“装配图 + B-Rep 几何”的图网络，预测两个零件之间是否需要 joint，以及 joint 元素；
- 使用 Fusion 360 Gallery Assembly Dataset 做训练与评测，达到接近人类水平的 joint 预测准确率（约 79.5%，接近人类 80%）。

**输入 / 输出**

- 输入：含多个零件的 CAD 装配（含 B-Rep 曲面和装配图）；
- 输出：关节预测列表（joint 类型 + 参数），即一个“装配约束方案”。

> 对你来说：JoinABLe 提供了一个完整 pipeline 的示范：**用真实装配数据 → 学 joint → 用于装配/运动学预测**。你可以复用其数据和模型结构，将 LLM 作为 planner，把 joint prediction 作为一个子模块。

#### 3.2.3 Mates2Motion（Learning How Mechanical CAD Assemblies Work）

- **是否开源**：论文 + 一部分代码 & 数据（用于非商业研究）  

**主要功能**

- 使用深度学习从 CAD 装配 + mates（约束）中推断实际的 **运动自由度**；
- 重新定义 mates，使之更匹配真实机械运动（例如多个约束组合形成一个等效转动副）。

**与 agent 相关的启发**

- 你可以将 “joint prediction”（JoinABLe） + “DOF 推断”（Mates2Motion）串成两级：
  - 一级：根据几何预测边上是否有 joint；
  - 二级：对所有 joint 进行聚合，推断机构总体 DOF，与经典机构学公式对比。

---

### 3.3 PartNet-Mobility / SAPIEN / ManiSkill（偏机器人侧的装配 & 运动学环境）

#### 3.3.1 PartNet-Mobility + SAPIEN

- **是否开源**：数据和环境可公开用于非商业研究（需同意对应条款）  

**内容**

- PartNet-Mobility：在 PartNet 对象的基础上，提供了带 **运动关节** 的 3D 对象（门、抽屉、机器人部件等）；
- SAPIEN：集成 PhysX 的仿真环境，可加载这些带关节的物体进行交互 / 机器人操作。

**用途**

- 可作为 **“装配后运动学仿真”环境**：
  - LLM/Agent 输出装配方案或 joint 参数 → SAPIEN 做真实物理仿真 → 返回反馈（能否顺畅运动、是否卡死）；
- 在文献中已被用于评估通用关节建模 & manipulation 能力。

#### 3.3.2 ManiSkill / IsaacLab 等（加载 PartNet-Mobility）

- ManiSkill 支持直接加载 PartNet-Mobility 等数据集中已有的关节信息，用于关节控制与任务训练。

> 对你来说：这类环境可以被视为“装配后的运动学 + 机器人执行验证平台”，适合评估 Agent 输出的 joint / DOF 是否合理。

---

## 4. 评测环境（执行 & 验证装配 / 运动学）

1. **FreeCAD Assembly + Python 脚本**
   - 使用 A2plus / 官方 Assembly Workbench 建立装配约束；
   - 编写脚本：
     - 自动读取 Agent 预测的关节 / 约束；
     - 调用约束求解器；
     - 对选定关节施加旋转 / 平移，记录是否：
       - 约束不可解；
       - 出现几何干涉（可利用第 3 步中的布尔运算模块，和 Step 3 复用环境）。
   - 指标：
     - 成功求解率；
     - 模拟运动成功率（无干涉，无奇异锁死）。

2. **Fusion 360 Gallery Assembly Dataset + JoinABLe 评测脚本**
   - 使用 JoinABLe 提供的代码和 Fusion 360 joint ground truth 数据：
     - 对关节存在性 / 类型分类进行准确率评估；
     - 用 LLM/Agent 输出的 joints 替换 JoinABLe 的 network 输出，对比精度。

3. **PartNet-Mobility / SAPIEN / ManiSkill 仿真**
   - 在已有带关节的模型基础上：
     - 将 Agent 输出的运动指令（如“旋转门 30°”、“拉出抽屉 10cm”）传入物理引擎；
     - 观察是否与预期 DOF 对齐（例如只绕某一轴转动，而不是房间里乱飞）。
   - 指标：
     - 成功完成预定运动的比例；
     - 运动过程中无碰撞 / 稳定性。

4. **A 3D CAD Assembly Benchmark（偏装配检索 / 结构对比）**

- 这是一个专门为 3D CAD 装配检索设计的 benchmark，聚焦齿轮箱等机械装配的结构相似性评估。  
- 虽然它不直接提供 joints DOF，但它提供了 **真实装配模型集**，可被你用于：
  - 生成“参考装配方案”；
  - 对比 Agent 输出的装配结构与 benchmark 中的 ground truth 装配结构。

---

## 5. 若要训练 Agent：可用的开源 / 公开训练数据集

### 5.1 Fusion 360 Gallery Assembly Dataset

- **用途**：
  - 训练模型从 B-Rep + 装配图 → 预测 joints；
  - 进一步推断 DOF、装配结构类别。
- **具体数据形式**（以 Joint Data 为例）：
  - `joint.json` 文件中记录 joint 类型、参与零件、局部坐标系；
  - `part.obj` / B-Rep 数据提供零件几何。

**训练样例构造示例**

- 输入：
  - 零件对 (part A, part B) + 接触面 / 孔集合；
- 输出：
  - 关节类型（枚举：rigid / revolute / slider / cylindrical / …）；
  - 关节轴线在全局坐标中的参数。

---

### 5.2 PartNet / PartNet-Mobility / GAPartNet 系列

- **PartNet-Mobility**：可模拟带关节的 3D 物体集合；在 SAPIEN 中可直接加载、仿真。  
- **GAPartNet**：在 PartNet-Mobility 等基础上，增加了更细粒度的交互部件标注。  

**如何用于装配 / 运动学训练**

- 将每个带关节的物体视作“小型装配”：
  - 关节参数 = ground truth；
  - Agent/模型任务：
    - 根据几何和语义标注，预测哪些部件之间有 joint；
    - 预测 joint 类型与运动范围；
- 与机械装配相对，更多分布在家居 / 人机互动类物体，但对“运动副 + DOF” 学习很有价值。

---

### 5.3 Assembly101（多视角人类装配视频）

- **是否开源**：视频数据 + 标注可公开下载，用于研究  
- **数据内容**：
  - 4321 段装配/拆卸玩具车的视频；
  - 8 个静态视角 + 4 个头戴视角（多视角同步）；
  - 100k+ 粗粒度动作段、100 万+ 细粒度动作段，18M 3D 手部姿态标注。  

**与本步骤的关系**

- 可用于训练 Agent 学习“装配顺序 / 动作序列”，结合 CAD 几何环境：
  - 例如：从视频 + 文本描述 → 推断装配顺序；
  - 然后在 CAD/FreeCAD 环境中，验证顺序是否导致可行的装配（不会出现先装入被挡住的零件）。

---

## 6. 若要评测 Agent：可用的开源 / 公开评测集

### 6.1 JoinABLe 官方评测（Fusion 360 Gallery Assembly Dataset）

- **评测任务**：
  - 关节存在性预测；
  - 关节类型分类；
  - 关节参数回归（轴线等）。
- **指标**：
  - 关节分类准确率（paper 报告约 79.5%，接近人工水平）；
  - 在真实装配上的性能，可作为 Agent 输出 joint 的对比基线。

> 你可以复用 JoinABLe 的评测脚本，把 LLM/Agent 的输出接入同一评测流水线。

---

### 6.2 SAPIEN / PartNet-Mobility 运动评测

- **评测任务**：
  - 对给定关节控制指令，评估 Agent 提供的运动学建模是否能在物理仿真中正确执行；
- **指标**：
  - 任务成功率（如开门 / 拉抽屉 / 关柜门等）；
  - 与 ground truth joint 参数的偏差（如转角误差、位移误差）。

---

### 6.3 A 3D CAD Assembly Benchmark

- **用途**：
  - 评估装配结构相似性检索能力——例如：
    - 给定一个减速器装配，检索数据库中结构相似的装配；
  - 可作为“高层装配拓扑”评测数据集。
- **对 Agent 的评测方式**（建议）：
  - Agent 读取单个装配 → 输出抽象装配图（节点 = 功能零件，边 = 接触/关节类型）；
  - 与 benchmark 提供的 ground truth 装配图做对比（图匹配指标 / 检索准确率）。

---

## 7. 小结 & 对你构建 Agent 的启示

- 装配 / 运动学这个 Link 上，**现有资源是“数据 + 原生数值方法 + 部分 DL 模型”组合，而不是一个现成的 LLM agent**：
  - 数值装配与运动学的“执行环境”：FreeCAD Assembly、OCCT、SAPIEN/ManiSkill 等已经很成熟；
  - joint / motion 学习：JoinABLe、Mates2Motion、PartNet-Mobility 生态提供了大规模真实或可模拟的数据；
  - 人类装配过程：Assembly101 提供了步骤层面的序列数据。
- 如果你要系统性评估不同 LLM 在这一环节的能力，可以按以下思路设计 benchmark：
  1. **结构理解侧**：用 Fusion 360 Gallery Assembly + JoinABLe 的 joint ground truth 测试“关节识别 & DOF 推断”；
  2. **执行侧**：把 Agent 输出的装配 / joints 投入 FreeCAD / SAPIEN 进行实际运动仿真，考察可行性与稳定性；
  3. **过程侧**：结合 Assembly101 的装配顺序，评估 Agent 是否能给出合理的“装配规划 + 运动顺序”。

> 这样，你的“Step 4 – 装配 / 运动学”模块就既有：  
> - 真实 CAD 装配数据（Fusion 360, A 3D CAD Assembly Benchmark），  
> - 运动学 ground truth（PartNet-Mobility, JoinABLe, Mates2Motion），  
> - 执行环境（FreeCAD, SAPIEN/ManiSkill），  
> 可以比较干净地拆成：**装配图推理 → joint/DOF 预测 → 仿真验证** 三层来评测不同大模型 / Agent 框架。

