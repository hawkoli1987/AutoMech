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

## Available Demos

### 1. Data Loader (`demo_data_loader.py`)

Shows how to load samples from the LLM4CAD dataset.

```bash
python3 scripts/demo_data_loader.py
```

**What it demonstrates:**
- Loading samples from all 5 categories (Flange, Gear, Nut, Shaft, Spring)
- Displaying text descriptions and parameter specs
- Converting samples to GraphState for agent processing

### 2. LLM Client (`demo_llm_client.py`)

Shows how to use the LLM client for text and structured generation.

```bash
python3 scripts/demo_llm_client.py
```

**What it demonstrates:**
- Simple text generation
- JSON extraction from responses
- Structured output with Pydantic models
- Chat with system prompts

**Requires:** Qwen3-8B running on port 8001

### 3. VLM Client (`demo_vlm_client.py`)

Shows how to use the VLM client for image-based analysis.

```bash
python3 scripts/demo_vlm_client.py
```

**What it demonstrates:**
- Image description
- JSON extraction from image analysis
- CAD quality evaluation (simulating VLM judge)

**Requires:** Qwen3-VL running on port 8002

### 4. Design Agent (`demo_design_agent.py`)

Shows the LangGraph design agent structure (with mocked LLM calls).

```bash
python3 scripts/demo_design_agent.py
```

**What it demonstrates:**
- Agent architecture and node flow
- Category schemas for each part type
- Heuristic parameter scoring
- State management and termination logic

### 5. Storage Layer (`demo_storage.py`)

Shows how to use the SQLite storage layer.

```bash
python3 scripts/demo_storage.py
```

**What it demonstrates:**
- Database initialization
- Sample and run management
- Node execution logging
- Artifact tracking
- Statistics queries

## Running Tests

Use the test runner script:

```bash
# Unit tests only (no live LLM/VLM)
./scripts/run_tests.sh

# All tests including live tests
./scripts/run_tests.sh --live

# Verbose output
./scripts/run_tests.sh -v --live
```

Or run pytest directly:

```bash
# All tests
python3 -m pytest tests/ -v

# Specific component
python3 -m pytest tests/test_data_loader.py -v

# Live tests only
python3 -m pytest tests/test_llm_client.py::TestLLMClientLive -v
python3 -m pytest tests/test_llm_client.py::TestVLMClientLive -v
```

