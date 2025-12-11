# Agentic Parametric CAD Design Framework
*Automated parametric CAD generation, evaluation, and simulation via LangGraph, LLMs, VLMs, and CAD/FEA tools.*

## Overview
This project implements an agentic framework for mechanical part design, driven by parametric CAD and multimodal reasoning.  
The system ingests data from the LLM4CAD dataset (text, image, parametric CAD specification), generates new parametric designs, iteratively evaluates them using LLM/VLM judges, synthesizes CAD models via CadQuery/FreeCAD, and optionally performs FEA simulation using Calculix.

It is designed for:
- Automated CAD generation  
- Parametric specification synthesis  
- Iterative agentic optimization  
- 2D/3D consistency judging  
- FEA-based physical evaluation  
- Dataset creation and ML research

The architecture is modular, reproducible, and extensible to additional CAD/CAE tools or MCP-based tool unification.

---

## Architecture

The system consists of three major layers:

```
Dataset Loop  →  Design Agent (LangGraph)  →  FEA Agent (LangGraph Subgraph)
```

---

## 1. Dataset Layer (Batch Processing Loop)

A Python-based loop iterates through the LLM4CAD triplets:

- text_desc (natural-language description)
- image_paths (rendered CAD images)
- gt_param_spec (ground-truth parametric design specification)

For each sample:
1. Construct the initial GraphState  
2. Invoke the Design Agent  
3. Persist all outputs (CAD files, images, scores)  
4. Log the full trace into a structured process database  
5. Optionally append results into a DataFrame or Parquet file  

The Dataset Loop contains no agentic logic; it manages I/O, batching, metadata, and high-level orchestration.

---

## 2. Design Agent (LangGraph)  
### Single-component iterative design + multimodal evaluation

The Design Agent acts as a closed-loop optimization system. It generates a parametric specification, validates it, synthesizes a CAD model, evaluates it visually and semantically, then decides whether to iterate further.

---

### 2.1 GraphState

```
GraphState = {
    "text_desc": str,
    "gt_param_spec": dict,
    "pred_param_spec": dict or None,

    "param_score": float or None,
    "param_feedback": str or None,

    "cad_file": str or None,
    "render_image": str or None,

    "vlm_score": float or None,
    "vlm_feedback": str or None,

    "iteration": int,
    "done": bool,

    "metadata": dict
}
```

---

### 2.2 Nodes

#### GenerateParamSpec (DeepSeek-V3.2 LLM)
Produces a structured parametric design specification from text.

#### JudgeParamSpec (DeepSeek-V3.2 critic)
Evaluates the generated param specification against the ground truth. Outputs score + feedback.

#### GenerateCAD (CadQuery / FreeCAD)
Converts the predicted param spec into:
- 3D CAD model (STEP / STL / FCStd)
- Rendered image (PNG)

#### JudgeCAD_VLM (Qwen3-VL)
Evaluates geometric & semantic consistency between:
- Rendered CAD image  
- Parametric design specification  
- Text description (optional)

Outputs score + feedback.

#### DecideNextStep (rule-based or small LLM)
Decides whether to iterate or terminate the loop.

---

### 2.3 Control Flow

```
GenerateParamSpec
        ↓
JudgeParamSpec
        ↓
GenerateCAD
        ↓
JudgeCAD_VLM
        ↓
DecideNextStep ── if not done → go to GenerateParamSpec
```

---

## 3. FEA Agent (LangGraph Subgraph)

Triggered after a satisfactory design is produced.

### Nodes
#### GenerateFEASpec (DeepSeek-V3.2)
Produces FEA spec including loads, constraints, materials.

#### TranslateToCalculixInput
Converts fea_spec + cad_file into .inp format.

#### RunCalculix
Executes the solver and extracts stress/displacement fields.

#### SummarizeFEAResult (LLM)
Generates engineering insight from numeric outputs.

---

## Structured Storage Layer

A structured database stores every step for reproducibility.

### Tables

#### samples
Metadata per LLM4CAD sample.

#### runs
One row per agent run.

#### nodes
One row per node execution.

#### artifacts
Paths to CAD, images, FEA files, etc.

### File Storage Layout

```
artifacts/
    cad/
    images/
    fea_inputs/
    fea_outputs/
```

---

## Repository Structure

```
agentic-cad-framework/
│
├── data/
├── artifacts/
├── db/
│   └── process.sqlite
│
├── src/
│   ├── data_loop/
│   ├── agents/
│   ├── cad/
│   ├── fea/
│   ├── judges/
│   ├── storage/
│   └── utils/
│
├── configs/
├── docs/
└── requirements.txt
```

---

## Dependencies

### Core
- langgraph  
- langchain  
- transformers  
- openai  
- pydantic  

### CAD
- cadquery  
- ocp_tessellate  
- trimesh  
- freecad (system-level)

### FEA
- numpy  
- calculix-ccx (external binary)

### Utilities
- pandas  
- pyarrow  
- sqlalchemy  
- duckdb  
- pillow  
- matplotlib  
- rich  

## LLM Evalution Pipeline

### Dependency requirement

- Currently, most of dependencies are installed inside the container 'vllm_mech', which running inside a compute node with a single GPU H200. 
- The installation process can be found in `enroot_modify.sh`
- If you find any missing dependencies, install them into the container, and add the installation step into `enroot_modify.sh`

---

## Implementation Plan

### Phase 1: Foundation & Design Agent ✅ COMPLETED

#### Step 1.1: Project Scaffolding ✅
- [x] Create repository structure matching spec (`src/`, `configs/`, `artifacts/`, `db/`)
- [x] Create `requirements.txt` with pinned versions
- [x] Set up Pydantic schemas for `GraphState`, `ParamSpec`, `CADResult`, `JudgeResult`
- [x] Create configuration system (YAML-based) for LLM endpoints, thresholds, paths

#### Step 1.2: Data Loaders ✅
- [x] **LLM4CAD Loader**: Parse category folders → `(text_desc, json_params, stl_path, image_path)`
- [x] Batch iterator with configurable sample selection (max_samples_per_category, shuffle)
- [ ] **GenCAD-Code Loader**: Deferred (LLM4CAD is the primary dataset)

#### Step 1.3: LLM Client Abstraction ✅
- [x] Create `LLMClient` base class with `generate()`, `generate_json()`, `generate_structured()` methods
- [x] Implement `VLLMClient` for local vLLM (Qwen3-8B on port 8001)
- [x] Implement `VLMClient` for Qwen3-VL (port 8002) with image support
- [x] Add retry logic, timeout handling, JSON extraction from markdown
- [x] **Live tested** with actual Qwen3-8B and Qwen3-VL-8B-Instruct models

#### Step 1.4: LangGraph Design Agent ✅
- [x] Define `DesignAgentState` Pydantic model matching spec
- [x] **GenerateParamSpec Node**: Text → Structured JSON parameters
- [x] **JudgeParamSpec Node**: Heuristic scoring with LLM fallback
- [x] **GenerateCAD Node**: Params → CadQuery → STEP/STL + rendered PNG
- [x] **JudgeCAD_VLM Node**: VLM-based visual evaluation (stubbed, VLM client ready)
- [x] **DecideNextStep Node**: Rule-based termination logic
- [x] Wire nodes with conditional edges in LangGraph

#### Step 1.5: CAD Generation Module ✅
- [x] Template-based CadQuery generators for all 5 categories (Flange, Nut, Shaft, Gear, Spring)
- [x] Export to STEP and STL formats
- [x] Rendering pipeline with `ocp_tessellate` integration
- [x] Error handling for invalid geometry

#### Step 1.6: Storage Layer ✅
- [x] SQLite schema for `samples`, `runs`, `node_executions`, `artifacts`
- [x] SQLAlchemy ORM models with relationships
- [x] DatabaseManager singleton with CRUD operations
- [x] Statistics and analytics queries

#### Step 1.7: Dataset Processing Loop ✅
- [x] Main loop orchestrating: load sample → invoke agent → persist results
- [x] Progress tracking with `rich` console output
- [x] WandB integration for experiment tracking (optional)
- [x] Checkpoint/resume capability with JSON-based checkpoints

### Phase 2: VLM Integration & Refinement 🔄 IN PROGRESS

#### Step 2.1: VLM Judge ✅
- [x] Integrate Qwen3-VL for multimodal evaluation (port 8002)
- [x] Design VLM prompts for geometric consistency scoring
  - Enhanced system prompt with detailed scoring criteria
  - Structured JSON output with geometric, completeness, quality scores
  - Actionable feedback and issues list
- [x] Calibrate score thresholds (tuning in progress)
  - Current thresholds: param_score > 0.85, vlm_score > 0.8
  - Observed average VLM scores: ~0.58 on LLM4CAD images

#### Step 2.2: Iterative Refinement ✅
- [x] Implement feedback-driven re-generation in GenerateParamSpec
  - Combines param judge feedback + VLM feedback + previous prediction
  - Formatted as clear iteration improvement instructions
- [x] Add iteration history to GraphState
  - Tracks all scores, predictions, and feedback per iteration
  - Includes VLM detail scores (geometric, completeness, quality)
- [ ] Fine-tune termination criteria (pending end-to-end testing)

### Phase 3: FEA Agent (Subgraph)

#### Step 3.1: FEA Spec Generation
- [ ] Design `FEASpec` schema (loads, constraints, materials)
- [ ] LLM node to generate FEA spec from CAD + text description

#### Step 3.2: CalculiX Integration
- [ ] STEP → Mesh conversion (using Gmsh)
- [ ] Generate `.inp` files from FEA spec
- [ ] Execute CalculiX solver
- [ ] Parse output (stress, displacement fields)

#### Step 3.3: FEA Result Summarization
- [ ] LLM node to interpret numeric results
- [ ] Pass/fail criteria for structural requirements

---

## Verification Tests

### Unit Tests

| Test ID | Component | Description | Pass Criteria |
|---------|-----------|-------------|---------------|
| T1.1 | Data Loaders | Load 10 samples from LLM4CAD | All 5 categories parsed, no exceptions |
| T1.2 | Data Loaders | Load 100 samples from GenCAD-Code Parquet | Images decoded, code strings valid |
| T1.3 | LLM Client | Query Qwen3-8B with simple prompt | Response received within 30s |
| T1.4 | LLM Client | Structured output (JSON mode) | Valid JSON matching schema |
| T1.5 | CAD Generator | Generate Flange from params | Valid STEP file, non-zero volume |
| T1.6 | CAD Generator | Render STEP to PNG | PNG file exists, >10KB size |
| T1.7 | Storage | Insert and query run record | Record retrieved matches inserted |

### Integration Tests

| Test ID | Scenario | Description | Pass Criteria |
|---------|----------|-------------|---------------|
| T2.1 | Single Sample E2E | Run Design Agent on 1 LLM4CAD sample | All nodes execute, CAD file generated |
| T2.2 | Iteration Loop | Agent iterates at least 2 times before terminating | `iteration >= 2` in final state |
| T2.3 | Score Threshold | High-quality sample terminates in 1 iteration | `done=True` after first pass |
| T2.4 | Error Recovery | Invalid params → regeneration attempt | No crash, iteration count increases |
| T2.5 | Batch Run | Process 10 samples sequentially | All samples logged to DB |

### Validation Tests

| Test ID | Metric | Description | Pass Criteria |
|---------|--------|-------------|---------------|
| T3.1 | Param Accuracy | Compare predicted vs. GT params on 100 samples | Mean IoU > 0.7 for numeric params |
| T3.2 | CAD Validity | Generated STL passes mesh integrity check | No non-manifold edges, watertight |
| T3.3 | VLM Consistency | VLM score correlates with human judgment | Spearman ρ > 0.6 on 50 samples |

---

## Resolved Design Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | **Starting Dataset** | LLM4CAD (text→params, with STL meshes to render images) |
| 2 | **LLM Endpoint** | `http://localhost:8001` via `OPENAI_API_BASE` env var |
| 3 | **VLM Available** | Yes, Qwen3-VL deployed on same infrastructure |
| 4 | **Termination Thresholds** | `param_score > 0.85 AND vlm_score > 0.8` |
| 5 | **Max Iterations** | 5 iterations before forced termination |
| 6 | **FEA Scope** | Deferred to Phase 2/3 |
| 7 | **Model Flexibility** | Build own LLM serving abstraction following `inference/server` and `inference/client` patterns |

---

## LLM Client Architecture (Following inference/ patterns)

The LLM client will follow the existing patterns from `inference/client/` and `inference/server/`:

### Environment Variables
```bash
export OPENAI_API_BASE=http://localhost:8001
export OPENAI_API_KEY=dummy  # vLLM doesn't require auth
export OPENAI_MODEL=Qwen/Qwen3-8B  # Optional, auto-detected from /v1/models
```

### Client Pattern (from inference/client/rewrite.py)
```python
from openai import OpenAI

# Auto-detect model from vLLM server
client = OpenAI(base_url=os.getenv("OPENAI_API_BASE") + "/v1", api_key="dummy")
model_id = client.models.list().data[0].id

# Chat completion with Qwen3 specific params
response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    max_tokens=1024,
    temperature=0.7,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}  # Qwen3 specific
)
```

### Model Templates (from inference/templates/)
Store model-specific configs in JSON:
```json
{
  "model": "Qwen/Qwen3-8B",
  "dtype": "bfloat16",
  "max_model_len": 8192,
  "enable_prefix_caching": true
}
```

### Flexible Model Switching
The system will support switching between models by:
1. **Server-side**: Use `inference/server/serve_enroot.sh` patterns to start different models
2. **Client-side**: Auto-detect model ID from `/v1/models` endpoint
3. **Config-driven**: Template files define model-specific parameters

---

## LLM4CAD Data Schema Mapping

### Raw Data Format
```
LLM4CAD/
├── Flange/
│   ├── mesh/
│   │   └── flange_XXXXX.stl        # 3D mesh (1000 files)
│   ├── dimension_text/
│   │   └── flange_XXXXX.json       # Parametric specs
│   └── Flange_description.csv       # Text descriptions
├── Gear/
├── Nut/
├── Shaft/
└── Spring/
```

### Mapping to GraphState
| LLM4CAD Field | GraphState Field | Transform |
|---------------|------------------|-----------|
| CSV `answer` column | `text_desc` | Direct copy |
| JSON file | `gt_param_spec` | Parse JSON |
| STL path | `metadata.stl_path` | Store path, render to `render_image` |
| Category | `metadata.category` | Extract from path |

### Example Data Sample
```python
@dataclass
class LLM4CADSample:
    sample_id: str           # "flange_00001"
    category: str            # "Flange"
    text_desc: str           # "The object is a flange with diameter 124mm..."
    gt_param_spec: dict      # {"base_diameter": 124, "base_height": 19, ...}
    stl_path: Path           # Path to STL file
    
    def to_graph_state(self) -> GraphState:
        return GraphState(
            text_desc=self.text_desc,
            gt_param_spec=self.gt_param_spec,
            pred_param_spec=None,
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            metadata={"sample_id": self.sample_id, "category": self.category, "stl_path": str(self.stl_path)}
        )
```

---

## Prompt Templates

### GenerateParamSpec Prompt
```
SYSTEM: You are a mechanical engineer assistant that extracts parametric specifications from natural language descriptions.

USER: Given the following description of a mechanical part, extract the parametric specification as a JSON object.

Description: {text_desc}
Category: {category}

Expected parameters for {category}:
{parameter_schema}

Output only the JSON object, no explanation.
```

### JudgeParamSpec Prompt
```
SYSTEM: You are a CAD parameter judge. Compare predicted parameters against ground truth and score the accuracy.

USER: Compare these two parameter specifications:

Ground Truth:
{gt_param_spec}

Predicted:
{pred_param_spec}

Score each parameter:
- Exact match: 1.0
- Within 5% tolerance: 0.8
- Within 10% tolerance: 0.6
- Larger deviation: 0.0

Output JSON:
{
  "overall_score": <float 0-1>,
  "parameter_scores": {<param>: <score>, ...},
  "feedback": "<specific issues>"
}
```

### JudgeCAD_VLM Prompt
```
SYSTEM: You are a CAD visual quality judge. Evaluate if the rendered 3D model matches the specification.

USER: [IMAGE]

Evaluate this rendered CAD model against the specification:
{pred_param_spec}

Original description: {text_desc}

Score on:
1. Geometric correctness (0-1)
2. Feature completeness (0-1)
3. Overall visual quality (0-1)

Output JSON:
{
  "overall_score": <float 0-1>,
  "geometric_score": <float>,
  "completeness_score": <float>,
  "quality_score": <float>,
  "feedback": "<visual issues observed>"
}
```

---

## Implementation Deviations & Design Decisions

### Data Format Handling
| Aspect | Original Assumption | Actual Implementation |
|--------|---------------------|----------------------|
| **Shaft Parameters** | Dictionary format like other categories | **List of [diameter, length] pairs** - ShaftSpec uses `sections: List[List[float]]` |
| **gt_param_spec Type** | Always `dict` | **Union[Dict, List]** - to accommodate Shaft's list format |
| **Image Availability** | Assumed images in dataset | **Images exist in `img/` subdirectory** per category |

### LLM/VLM Configuration
| Aspect | Design Decision | Rationale |
|--------|-----------------|-----------|
| **Dual Port Setup** | LLM on 8001, VLM on 8002 | Separate vLLM servers for text and vision models |
| **Model Auto-Detection** | Query `/v1/models` endpoint | Avoids hardcoding model names, flexible switching |
| **Qwen3 Thinking Mode** | Disabled by default | Faster responses for structured output tasks |
| **VLM Temperature** | Lower (0.3 vs 0.7) | More deterministic evaluation scores |

### Architecture Decisions
| Component | Decision | Rationale |
|-----------|----------|-----------|
| **JudgeParamSpec** | Heuristic scoring first, LLM fallback | Fast initial scoring, LLM for refinement feedback |
| **CAD Generators** | Template-based per category | Reliable geometry vs. LLM-generated code risks |
| **Database** | Naive datetime (no timezone) | SQLite compatibility, simpler storage |
| **Checkpointing** | JSON file per run | Simple, human-readable, resume capability |

### Test Coverage
- **124 tests passing**, 23 skipped (CadQuery-dependent)
- **8 live tests** with actual LLM/VLM servers
- Tests organized by component: scaffolding, data_loader, llm_client, design_agent, cad_generators, storage, runner

---

## Next Steps (Phase 2)

1. **Integrate VLM Judge fully** - Replace stub with actual Qwen3-VL calls
2. **Calibrate scoring thresholds** - Run on sample batch, analyze score distributions
3. **Implement feedback-driven regeneration** - Use judge feedback in next iteration
4. **Add CadQuery execution environment** - Run inside container with CadQuery installed
5. **End-to-end validation** - Process 100 samples, measure param accuracy

---

## Running the Framework

### Quick Start Scripts

See `scripts/` folder for component demos:

```bash
# Inside tmux 'dev' session
cd /scratch/Projects/SPEC-SF-AISG/source_files/AutoMech

# Demo: Data Loader
python3 scripts/demo_data_loader.py

# Demo: LLM Client
python3 scripts/demo_llm_client.py

# Demo: VLM Client
python3 scripts/demo_vlm_client.py

# Demo: Design Agent (mocked)
python3 scripts/demo_design_agent.py

# Run all tests
python3 -m pytest tests/ -v
```

### Environment Setup

```bash
export OPENAI_API_BASE=http://localhost:8001   # Qwen3-8B
export OPENAI_API_BASE2=http://localhost:8002  # Qwen3-VL
```
