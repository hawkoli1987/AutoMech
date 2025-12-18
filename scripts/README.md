# Demo Scripts

Interactive demonstrations of the Freeform CAD Generation Framework.

**Note**: This framework has been refactored to focus exclusively on **freeform CAD generation** using LLM-generated CadQuery code. Fixed category templates (Flange, Gear, Nut, Shaft, Spring) have been removed.

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

**Purpose**: Create samples from text descriptions for freeform CAD generation.

```bash
python3 scripts/demo_data_loader.py
```

**User Input Required**: None (uses predefined examples)

**What It Does**:
1. Creates `LLM4CADSample` objects from text descriptions
2. Shows how to convert samples to `GraphState` for agent processing
3. Demonstrates the simplified data loading approach for freeform generation

**Expected Output**:
```
╭───────────────────────────────────────╮
│ Freeform CAD Sample Creation Demo    │
╰───────────────────────────────────────╯

Creating Samples from Text Descriptions:
┌──────────────────────┬────────────────────────────────────────────────┐
│ Sample ID            │ Text Description                               │
├──────────────────────┼────────────────────────────────────────────────┤
│ freeform_a1b2c3d4    │ A mounting bracket with two 8mm holes...       │
│ freeform_e5f6g7h8    │ A cylindrical spacer with outer diameter...    │
└──────────────────────┴────────────────────────────────────────────────┘

✓ Sample creation demo complete!
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

✓ LLM client demo complete!
```

---

### 3. VLM Client (`demo_vlm_client.py`)

**Purpose**: Test VLM image description and structured analysis.

```bash
python3 scripts/demo_vlm_client.py
```

**User Input Required**: None (uses images from artifacts)

**What It Does**:
1. Connects to Qwen3-VL on port 8002
2. Finds a sample PNG image
3. **Image Description**: Asks VLM to describe the mechanical part
4. **JSON Analysis**: Extracts structured info (part_type, features, complexity)
5. **CAD Evaluation**: Simulates the VLM judge scoring the render

**Expected Output**:
```
╭────────────────╮
│ VLM Client Demo│
╰────────────────╯

Using image: artifacts/renders/test_freeform_spacer.png

1. Initializing VLM Client...
   Model: Qwen/Qwen3-VL-8B-Instruct

2. Image Description
   Response: This image shows a cylindrical spacer...

✓ VLM client demo complete!
```

---

### 4. Design Agent (`demo_design_agent.py`)

**Purpose**: Demonstrate the LangGraph design agent structure for freeform generation.

```bash
python3 scripts/demo_design_agent.py
```

**User Input Required**: None

**What It Does**:
1. Shows the agent architecture (nodes and edges)
2. Displays the freeform code design schema (`CadQueryCodeDesign`)
3. Shows example generated code
4. **State Management**: Shows GraphState initialization and iteration
5. **Termination Logic**: Checks thresholds to decide if agent should iterate
6. **Graph Compilation**: Builds the LangGraph (without running full inference)

**Expected Output**:
```
╭─────────────────────────────────────────╮
│ Design Agent Demo - Freeform CAD Gen    │
╰─────────────────────────────────────────╯

1. Agent Architecture
   DesignAgent (LangGraph)
   ├── Nodes
   │   ├── generate_code - LLM generates CadQuery code from text
   │   ├── generate_cad - Execute code safely to create CAD
   │   └── judge_cad_vlm - VLM evaluates visual quality
   └── Flow
       └── generate_code → generate_cad → judge_cad_vlm → decide_next_step

2. Freeform Code Design Schema
   CadQueryCodeDesign: LLM-generated Python code for arbitrary parts

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
2. **Insert Samples**: Adds sample records
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
   Inserted: bracket_00001

3. Create Runs
   Created run: abc123... for bracket_00001

✓ Storage demo complete!
```

---

### 6. Freeform CAD Generation (`demo_freeform_cad.py`)

**Purpose**: End-to-end demo of Text → LLM → CadQuery Code → CAD files.

```bash
# Interactive mode (prompts for text description)
python3 scripts/demo_freeform_cad.py

# Run with predefined examples
python3 scripts/demo_freeform_cad.py --examples
```

**User Input**: 
- **Interactive mode** (default): User enters component description
- **--examples**: No input, uses predefined examples

**What It Does**:
1. **Check Dependencies**: Verifies LLM server and CadQuery availability
2. **Text → Code**: LLM generates complete CadQuery Python code
3. **Code Validation**: Safety checks (no file I/O, no dangerous operations)
4. **Code Execution**: Safely executes code in sandboxed environment
5. **CAD Export**: Generates STEP/STL files and PNG renders
6. **Output Files**: Saves to `artifacts/cad/` and `artifacts/renders/`

**Expected Output** (interactive):
```
╭───────────────────────────────────────╮
│ Freeform CAD Code Generation Demo    │
╰───────────────────────────────────────╯

Example Descriptions:
  • A mounting bracket with two holes and a curved support arm
  • A T-joint connector with 20mm diameter arms
  • A custom spacer ring with outer diameter 40mm, inner 25mm

Enter mechanical part description: A cylindrical spacer with outer diameter 30mm

Step 1: Generating CadQuery Code
→ Calling LLM to generate code...
✓ Code generated successfully

Step 2: Code Validation
┌──────────────────┬────────┬─────────┐
│ Check            │ Status │ Details │
├──────────────────┼────────┼─────────┤
│ Syntax Check     │ ✓ PASS │ OK      │
│ Security Check   │ ✓ PASS │ OK      │
│ Length Check     │ ✓ PASS │ OK      │
└──────────────────┴────────┴─────────┘

✓ Code passed all safety checks

Step 3: Executing Code Safely
✓ Code executed successfully

Step 4: Exporting CAD Files
✓ STEP file: artifacts/cad/demo_freeform.step
✓ STL file: artifacts/cad/demo_freeform.stl
✓ Render PNG: artifacts/renders/demo_freeform.png

✓ Freeform CAD Generation Complete!
```

**Note**: CadQuery must be installed (`pip install cadquery`).

---

### 7. VLM Judge (`demo_vlm_judge.py`)

**Purpose**: Test the VLM judge with actual CAD images.

```bash
python3 scripts/demo_vlm_judge.py
```

**User Input Required**: None (uses images from artifacts)

**What It Does**:
1. Connects to Qwen3-VL on port 8002
2. Loads sample images
3. Evaluates geometric accuracy, completeness, and quality
4. Provides detailed scores and feedback

**Expected Output**:
```
┌────────────────┬───────┬───────────┬────────┐
│ Score Type     │ Value │ Threshold │ Status │
├────────────────┼───────┼───────────┼────────┤
│ Overall        │ 0.850 │ 0.80      │ PASS   │
│ Geometric      │ 0.900 │ 0.80      │ PASS   │
│ Completeness   │ 0.800 │ 0.80      │ PASS   │
│ Quality        │ 0.850 │ 0.80      │ PASS   │
└────────────────┴───────┴───────────┴────────┘

Feedback: Geometry matches description well...
```

---

## Running Tests

```bash
# Unit tests only (no LLM/VLM needed)
python3 -m pytest tests/ -v

# All tests including live LLM/VLM tests
python3 -m pytest tests/ -v --live

# Specific test file
python3 -m pytest tests/test_cad_generators.py -v
```

---

## Key Changes from Previous Version

**Removed**:
- Fixed category templates (Flange, Gear, Nut, Shaft, Spring)
- Category-specific parameter extraction
- `demo_cad_generator.py` (replaced by `demo_freeform_cad.py`)
- `LLM4CADLoader` with category-specific logic

**Added/Enhanced**:
- Freeform CadQuery code generation
- Safety validation for LLM-generated code
- Sandboxed code execution
- Simplified data loading (`create_sample_from_text`)

**Philosophy**:
The framework now focuses on **truly flexible parametric design** where the LLM generates complete CadQuery code for arbitrary mechanical parts, rather than being constrained to predefined categories.
