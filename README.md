## 统一 Benchmark 资源整理

### 1. 需求抽取

- **数据**：LLM4CAD / CAD-Llama 文本 + 参数  
- **模型**：CAD-Llama, GPT-4o, Kimi K2 等通用 LLM

---

### 2. 参数化 CAD 生成

- **模型**：CAD-Llama, CAD-Assistant, CAD-Coder  
- **数据/基准**：LLM4CAD, CAD-Llama datasets  
- **环境**：CadQuery, FreeCAD

### 3. CAD/FEA Tool-Use

- **模型**：CAD-Assistant / LLM4CAD 风格 pipeline  
- **数据**：上述 CAD 任务集  
- **环境**：FreeCAD / CadQuery + FEA solver，封装成 Tool-Server

---

### 3. 几何 / 干涉推理

- **模型**：CAD-Assistant, CAD-MLLM 系列  
- **数据**：LLM4CAD mechanical components, SCOPE  
- **环境**：OCCT / FreeCAD 布尔运算

---

### 4. 装配 / 运动学

- **模型**：CAD-Assistant, CADialogue（对话+建模）  
- **数据**：公共机械装配 CAD（GrabCAD 等）+ 自建标注  
- **环境**：FreeCAD Assembly workbench

---

### 5. 载荷 / 结构估算

- **模型**：通用工程能力强的 LLM  
- **数据**：结构力学教材例题、自建题库  
- **环境**：SymPy + FreeCAD FEM / CalculiX

---



---

### 7. 设计优化 / 多目标

- **模型**：通用 LLM + RL/BO 控制器；参考 Cadrille  
- **数据**：从 LLM4CAD/CAD-Llama 选典型零件，自建优化任务  
- **环境**：CAD+FEA 自动 pipeline

---

### 8. 制造性 / BOM / 文档

- **模型**：通用 LLM + CAD-aware LLM  
- **数据**：开源硬件项目 + 自建 DFM/BOM 标注  
- **环境**：半自动脚本 + 人工审查

---

### 9. 综合评测

- **理论框架**：LLM4CAD survey + CAD-Llama + CAD-Assistant

---

> **你的任务是把上面零散资源整合成一个统一 benchmark。**