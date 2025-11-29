# Step 2：单零件参数化建模（Parametric Part Modeling）

这是机械设计自动化最核心的步骤之一，涉及从结构化参数或自然语言中生成可执行的 CAD 参数化建模脚本，并可通过 FreeCAD / CadQuery / OCC 环境进行验证。

本步骤关注如何从结构化需求（Step 1 输出）或自然语言描述中，生成参数化 CAD 脚本，并在实际 CAD 环境中执行验证。

## 1. 步骤定义与成功标准

### 1.1 定义

给定输入：

- Step 1 的结构化设计约束（如尺寸、孔位、功能、材料等）
- 或自然语言描述 + 可选草图 / 图片

任务是生成：

- 可执行的参数化建模代码（如 CadQuery、FreeCAD Python、OpenSCAD、Fusion360 FeatureScript）
- 其运行结果应为 3D 可编辑模型（STEP / STL / B-Rep）

该步骤是连接"语言/需求"与"几何实体"的关键。

### 1.2 成功标准

一个可用的 Agent 应满足：

**代码可执行率（Execution Success Rate）**

- 生成脚本能被 CadQuery/FreeCAD 环境顺利运行，不报错。

**几何合规性（Geometric Compliance）**

- 模型尺寸、孔位位置、拓扑结构满足输入的 JSON 约束；
- 可通过测量、布尔检查、体积/包络等方式验证。

**参数化正确性（Parametric Validity）**

- 修改输入 JSON（如改变厚度）后，模型仍能基于同一脚本重新生成，证明其"参数化设计"而非"硬编码"。

**工程可用性**

- 零件的基本工程特征合理，例如壁厚非负、孔径不超界、草绘约束可求解等。

## 2. 具体案例示例

### 2.1 输入（来自 Step 1 的 JSON）

```json
{
  "component_type": "L_bracket",
  "dimensions_mm": {
    "leg1": [100, 80, 10],
    "leg2": [80, 80, 10]
  },
  "mounting_holes": {
    "thread": "M6",
    "spacing_mm": [60, 40],
    "count": 4
  }
}
```

### 2.2 输出（期望的 CadQuery 脚本示例）

```python
import cadquery as cq

# Unpack parameters
leg1_L, leg1_W, leg1_T = 100, 80, 10
leg2_L, leg2_W, leg2_T = 80, 80, 10
spacing_x, spacing_y = 60, 40

# Base L shape body
leg1 = cq.Workplane("XY").box(leg1_L, leg1_W, leg1_T)
leg2 = cq.Workplane("XY").workplane(offset=leg1_T).box(leg2_L, leg2_W, leg2_T)

bracket = leg1.union(leg2.translate((0, 0, leg1_T)))

# Drill M6 holes
holes = [(+spacing_x/2, +spacing_y/2),
         (+spacing_x/2, -spacing_y/2),
         (-spacing_x/2, +spacing_y/2),
         (-spacing_x/2, -spacing_y/2)]

bracket = bracket.faces(">Z").workplane().pushPoints(holes).hole(6)

result = bracket
```

### 2.3 几何验证

- 检查模型包络尺寸是否与 JSON 匹配；
- 检查孔位与直径；
- 检查法向方向（L 型结构是否正确）；
- 检查体积、质心等信息是否合理。

## 3. 可用的模型 / 软件（真实验证）

本节列出可用于 Step 2 的"建模模型"与"CAD 环境"，并逐条验证其真实性与用途。

### 3.1 专门面向"CAD 参数化建模"的开源模型 / 数据资源

#### 3.1.1 CAD-Coder（MIT–IBM, 2025）

**是否开源：** 是（Apache-2.0）

**资源：**

- 论文：《CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation》
  - https://arxiv.org/abs/2503.05130
- GitHub（模型与代码）：
  - https://github.com/anniedoris/CAD-Coder

**主要功能：**

- 从图像 → CadQuery 代码
- 专门用于生成可执行参数化 CAD 脚本

**输入形式：**

- CAD 渲染图像
- 可附加文字说明

**输出形式：**

- Python CadQuery 建模脚本

**适用原因（Step 2）：**

- 是目前最"正宗"的开源 Text/Image → 参数化代码模型
- 可作为 Step 2 的强 baseline 或训练数据来源

**数据示例（来自官方仓库结构）：**

```json
{
  "image": "gencad_im/part_001230.png",
  "cadquery_script": "result = cq.Workplane('XY').rect(20,10).extrude(5)"
}
```

#### 3.1.2 GenCAD-Code 数据集（开源）

**是否开源：** 是（Hugging Face dataset）

**资源：**

- https://huggingface.co/datasets/anniedoris/GenCAD-Code

**内容：**

约 163k 三元组：

- CAD 渲染图像
- 对应 CadQuery 脚本
- 生成的网格
- 以及 real_photo_test（真实 3D 打印照片）

**用途：**

- 可用于训练模型将参数/文本转化为 CadQuery 脚本
- 也可从脚本中解析几何参数，为 Step 2 提供训练集

#### 3.1.3 LLM4CAD 多模态数据集

**是否开源：** 数据开放（Texas Data Repository）

**资源：**

- 数据集 DOI：https://doi.org/10.18738/T8/KV7HON

**内容：**

- 五类典型机械构件
- 包含文本描述、CAD 模型（STEP）、图片草图

**用途：**

- 可将文本描述与 CAD 模型对齐，用于训练 Text-to-Param
- 可用于评估几何合规性（模型生成结果是否符合语义描述）

#### 3.1.4 Fusion 360 Gallery Dataset（部分开放）

**是否开源：** 部分开放（数据开放，软件闭源）

**资源：**

- 论文：https://arxiv.org/abs/2003.10983

**内容：**

- 包含参数化建模序列、草图约束、重建任务

**用途：**

- 可用于训练 agent 的"参数化建模行为模仿"

## 4. 参数化 CAD 环境（执行建模代码 + 几何验证）

### 4.1 CadQuery（开源）

**是否开源：** 是

**资源：** https://github.com/CadQuery/cadquery

**主要功能：**

- Python API 的参数化建模工具
- 基于 OpenCascade

**输入：**

- Python 代码（参数化的 Workplane 组合）

**输出：**

- 内存中的 3D 模型
- 可导出 STEP/STL

**适用原因：**

- 执行 LLM 生成脚本，并进行几何测量
- 是当前研究界最常用的 Text-to-CAD 验证环境

### 4.2 FreeCAD（开源）

**是否开源：** 是（LGPL）

**资源：** https://www.freecad.org/

**功能：**

- 全功能参数化 CAD
- 支持 Python API 自动建模
- 可进行 Feature 级别的建模序列检查

**适用原因：**

- 更接近工业 CAD 设计流程
- 可做草绘约束检查、装配接口验证

### 4.3 OCC / PythonOCC（开源）

**是否开源：** 是（OCC LGPL）

**资源：** https://github.com/tpaviot/pythonocc-core

**功能：**

- OpenCascade 官方案例的 Python 封装
- 适合做精细几何检查（曲率、布尔运算等）

## 5. 若要训练 Agent：可用的开源训练数据集

用于 Step 2（从参数 → CAD / 从文本 → CAD）的训练集包括：

### 5.1 GenCAD-Code（开源，首选）

- 最适合训练"生成 CadQuery 代码"
- 提供大量脚本示例，可反推参数结构

### 5.2 LLM4CAD Dataset（研究开放）

- 提供多模态信息：文本描述、草图、STEP、CAD 程序
- 可配合 Step 1 结果构建 "JSON → CadQuery" 映射

### 5.3 Fusion 360 Gallery（部分开放）

- 提供真实工业零件的参数化建模序列
- 可用于 imitation learning

### 5.4 ABC Dataset（开源）

- 可用在几何合理性判别训练（合成 → 判别）

## 6. 若要评测 Agent：可用评测集与指标

### 6.1 开源评测集来源

- **GenCAD-Code 官方测试集**
  - 用于比较脚本执行率和几何误差

- **LLM4CAD 数据集**
  - 用于几何对齐和语义一致性验证

- **Fusion 360 Gallery Reconstruction Benchmarks**
  - 用于检查"可重建性与参数化正确性"

### 6.2 建议的评测指标（基于真实研究做法）

**代码执行率（Execution Success Rate）**

- 脚本是否能在 CadQuery/FreeCAD 中无报错运行

**几何误差（Geometric Error）**

- 使用点云距离 / Hausdorff 距离
- 或几何属性差异（体积、包络尺寸）

**参数传递正确率（Param Bind Accuracy）**

- 修改输入参数后，脚本是否能重新生成正确形状

**工程可行性检查**

- 是否存在负尺寸、不可求解草绘、Boolean 失败等问题
