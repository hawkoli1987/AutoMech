# Demo Scripts

Interactive demonstrations of each component in the Agentic CAD Framework.

## Prerequisites

Run all scripts from the project root directory, inside the `dev` tmux session:

```bash
cd /scratch/Projects/SPEC-SF-AISG/source_files/AutoMech
```

Set environment variables for LLM/VLM access:

```bash
export OPENAI_API_BASE=http://localhost:8001   # Qwen3-8B
export OPENAI_API_BASE2=http://localhost:8002  # Qwen3-VL-8B-Instruct
```

---

## Demo Scripts Detail

### 1. Data Loader (`demo_data_loader.py`)

**Purpose**: Load and browse samples from the LLM4CAD dataset.

```bash
python3 scripts/demo_data_loader.py
```

**User Input Required**: None (uses local data files)

**What It Does**:
1. Initializes the `LLM4CADLoader` with the local dataset
2. Shows category counts (Flange, Gear, Nut, Shaft, Spring)
3. Displays a table of 15 sample entries with IDs, categories, descriptions, and parameters
4. Loads a specific sample by ID (`Flange_00001`)
5. Demonstrates converting a sample to `GraphState` for agent processing
6. Shows the convenience function `load_llm4cad_samples()`

**Expected Output**:
```
╭──────────────────────────╮
│ LLM4CAD Data Loader Demo │
╰──────────────────────────╯

Category Counts:
  Flange: 5 samples
  Gear: 5 samples
  ...

Sample Preview:
┌────────────────┬────────┬──────────────────────────────────┬──────────────────────┐
│ Sample ID      │ Cat.   │ Text Description                 │ Params               │
├────────────────┼────────┼──────────────────────────────────┼──────────────────────┤
│ flange_00000   │ Flange │ The object is a flange with...   │ diameter=124, ...    │
└────────────────┴────────┴──────────────────────────────────┴──────────────────────┘

✓ Data loader demo complete!
```

---

### 2. LLM Client (`demo_llm_client.py`)

**Purpose**: Test LLM text generation, JSON extraction, and structured output.

```bash
python3 scripts/demo_llm_client.py
```

**User Input Required**: None (uses predefined prompts)

**What It Does**:
1. Connects to Qwen3-8B on port 8001
2. **Simple Generation**: Asks "What is a flange?" and shows the response
3. **JSON Generation**: Extracts dimensions from text as JSON dict
4. **Structured Output**: Attempts to parse LLM response into a Pydantic model
5. **Chat**: Multi-turn conversation with system prompt

**Expected Output**:
```
╭────────────────╮
│ LLM Client Demo│
╰────────────────╯

1. Initializing LLM Client...
   Model: Qwen/Qwen3-8B
   
2. Simple Text Generation
   Response: A flange is a protruding rim used to connect pipes...

3. JSON Generation
   Result: {'outer_diameter': 100.0, 'inner_diameter': 50.0, 'thickness': 10.0}

4. Structured Output (Pydantic)
   Result: FlangeParams(outer_diameter=120.0, inner_diameter=60.0, ...)
   
5. Chat with System Prompt
   Response: A shaft transmits torque while an axle...

✓ LLM client demo complete!
```

**Possible Errors**:
- **"Structured output error: Field required"**: The LLM returned JSON with different field names than expected. This happens when the LLM uses names like `"diameter"` instead of `"outer_diameter"`. The demo catches this and falls back to raw JSON extraction.

---

### 3. VLM Client (`demo_vlm_client.py`)

**Purpose**: Test VLM image description and structured analysis.

```bash
python3 scripts/demo_vlm_client.py
```

**User Input Required**: None (uses images from LLM4CAD dataset)

**What It Does**:
1. Connects to Qwen3-VL on port 8002
2. Finds a sample PNG image from the dataset
3. **Image Description**: Asks VLM to describe the mechanical part
4. **JSON Analysis**: Extracts structured info (part_type, features, complexity)
5. **CAD Evaluation**: Simulates the VLM judge scoring the render

**Expected Output**:
```
╭────────────────╮
│ VLM Client Demo│
╰────────────────╯

Using image: data/LLM4CAD/Gear/img/Gear_00994.png

1. Initializing VLM Client...
   Model: Qwen/Qwen3-VL-8B-Instruct

2. Image Description
   Response: This image shows a spur gear, a type of mechanical component...

3. Structured Analysis (JSON)
   Result: {'part_type': 'gear', 'features': ['teeth', 'hub'], 'estimated_complexity': 'medium'}

4. CAD Quality Evaluation
   Evaluation: {'visual_quality': 0.85, 'geometry_correctness': 0.9, 'feedback': '...'}

✓ VLM client demo complete!
```

---

### 4. Design Agent (`demo_design_agent.py`)

**Purpose**: Demonstrate the LangGraph design agent structure and scoring logic.

```bash
python3 scripts/demo_design_agent.py
```

**User Input Required**: None

**What It Does**:
1. Shows the agent architecture (nodes and edges)
2. Displays category schemas (Flange, Gear, Nut, Shaft, Spring)
3. **Heuristic Scoring**: Demonstrates how parameter scores are calculated
4. **State Management**: Shows GraphState initialization and iteration
5. **Termination Logic**: Checks thresholds to decide if agent should iterate
6. **Graph Compilation**: Builds the LangGraph (without running full inference)

**Expected Output**:
```
╭──────────────────────╮
│ Design Agent Demo    │
╰──────────────────────╯

1. Agent Architecture
   DesignAgent (LangGraph)
   ├── Nodes
   │   ├── generate_param_spec - LLM generates parameters from text
   │   ├── judge_param_spec - Score predicted vs ground truth
   │   └── ...
   └── Flow
       └── generate_param_spec → judge_param_spec → ... → decide_next_step

2. Category Schemas
   Flange: FlangeSpec
   Gear: GearSpec
   ...

3. Heuristic Param Scoring
   Exact match score: 1.000
   Close match score: 0.967
   Wrong match score: 0.333

4. Agent State
   Initial: iteration=0, done=False
   After iter 1: param_score=0.967, vlm_score=0.82

5. Termination Logic
   param_threshold: 0.85, vlm_threshold: 0.8
   should_terminate: True

✓ Design agent demo complete!
```

---

### 5. Storage Layer (`demo_storage.py`)

**Purpose**: Demonstrate SQLite database operations for run tracking.

```bash
python3 scripts/demo_storage.py
```

**User Input Required**: None (creates temporary database)

**What It Does**:
1. Creates a temporary SQLite database
2. **Insert Samples**: Adds sample records for Flange, Gear, Shaft
3. **Create Runs**: Creates agent run records linked to samples
4. **Log Nodes**: Records node execution times and results
5. **Log Artifacts**: Records file paths for CAD/render outputs
6. **Complete Run**: Marks run as finished with final scores
7. **Statistics**: Shows aggregate counts

**Expected Output**:
```
╭────────────────────╮
│ Storage Layer Demo │
╰────────────────────╯

1. Initialize Database
   Tables created: samples, runs, node_executions, artifacts

2. Insert Samples
   Inserted: flange_00001
   Inserted: gear_00001
   ...

3. Create Runs
   Created run: abc123... for flange_00001

4. Log Node Executions
   Logged: generate_param_spec (1523ms)
   Logged: judge_param_spec (892ms)

5. Log Artifacts
   Logged: cad_step (45KB)

6. Complete Run
   Run completed: success=True, param_score=0.95

7. Statistics
┌─────────────────────┬───────┐
│ Metric              │ Value │
├─────────────────────┼───────┤
│ Total Samples       │ 3     │
│ Total Runs          │ 2     │
│ Successful Runs     │ 1     │
└─────────────────────┴───────┘

✓ Storage demo complete!
```

---

### 6. CAD Generation Pipeline (`demo_cad_generator.py`)

**Purpose**: End-to-end demo of Text → LLM → Parametric Spec → CadQuery → CAD files.

```bash
python3 scripts/demo_cad_generator.py
```

**User Input Required**: None (uses predefined examples for all 5 categories)

**What It Does**:
1. **Check Dependencies**: Verifies LLM server and CadQuery availability
2. **Text → Param Spec**: LLM extracts parametric specs from text descriptions
3. **Param Spec → CAD**: CadQuery generates STEP files and PNG renders
4. **Summary**: Shows the complete pipeline flow

**Expected Output**:
```
╭───────────────────────────────╮
│ CAD Generation Pipeline Demo  │
╰───────────────────────────────╯

1. Checking Dependencies
   ✓ LLM server available
   ✓ CadQuery available (or ⚠ if not installed)

2. Text → Parametric Specification (via LLM)

   Flange
   Input: A circular flange with base diameter 120mm...
   Output: {"base_diameter": 120, "base_height": 15, ...}

   Gear
   Input: A spur gear with module 2.5, 24 teeth...
   Output: {"module": 2.5, "num_teeth": 24, ...}

   ... (all 5 categories)

3. Parametric Specification → CAD (via CadQuery)

┌──────────┬─────────┬────────────────────────┬──────────────┐
│ Category │ Status  │ STEP File              │ Volume (mm³) │
├──────────┼─────────┼────────────────────────┼──────────────┤
│ Flange   │ SUCCESS │ flange_demo_iter0.step │ 45678.0      │
│ Gear     │ SUCCESS │ gear_demo_iter0.step   │ 12345.0      │
└──────────┴─────────┴────────────────────────┴──────────────┘

4. Pipeline Summary
   [ASCII diagram of the pipeline]

✓ CAD generation demo complete!
Generated files are in: artifacts/demo/
```

**Note**: If CadQuery is not installed, the demo will show expected file paths but won't generate actual files.

---

### 7. VLM Judge (`demo_vlm_judge.py`)

**Purpose**: Test the VLM judge with actual CAD images from the dataset.

```bash
python3 scripts/demo_vlm_judge.py
```

**User Input Required**: None (uses images from LLM4CAD dataset)

**What It Does**:
1. Connects to Qwen3-VL on port 8002
2. Loads sample images from the dataset
3. Evaluates geometric accuracy, completeness, and quality
4. Provides detailed scores and feedback

**Expected Output**:
```
┌────────────────┬───────┬───────────┬────────┐
│ Score Type     │ Value │ Threshold │ Status │
├────────────────┼───────┼───────────┼────────┤
│ Overall        │ 0.670 │ 0.80      │ FAIL   │
│ Geometric      │ 0.700 │ 0.80      │ FAIL   │
│ Completeness   │ 0.500 │ 0.80      │ FAIL   │
│ Quality        │ 0.700 │ 0.80      │ FAIL   │
└────────────────┴───────┴───────────┴────────┘

Feedback: Verify the flange height and dimensions...
Issues Found:
  • The flange height appears shorter than specified
  • No chamfers or threads visible
```

---

## Running Tests

```bash
# Unit tests only (no LLM/VLM needed)
./scripts/run_tests.sh

# All tests including live LLM/VLM tests
./scripts/run_tests.sh --live

# Verbose output
./scripts/run_tests.sh -v --live
```

Or run pytest directly:

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_llm_client.py::TestLLMClientLive -v
```
