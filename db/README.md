# Database Schema

This directory contains the SQLite database for tracking agent runs, samples, and artifacts.

## File Location

```
db/process.sqlite
```

The database is created automatically when the pipeline first runs.

## Schema Overview

The database uses SQLite with 4 main tables that track the complete lifecycle of CAD generation runs.

```
┌──────────────┐     ┌──────────────┐
│   samples    │────<│     runs     │
└──────────────┘     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │  nodes   │  │ artifacts│  │  (logs)  │
       └──────────┘  └──────────┘  └──────────┘
```

## Table Definitions

### 1. `samples` - Dataset Samples

Stores metadata for each sample from the LLM4CAD dataset.

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR(64) | Primary key (e.g., "flange_00001") |
| `category` | VARCHAR(32) | Part category: Flange, Gear, Nut, Shaft, Spring |
| `text_desc` | TEXT | Natural language description |
| `gt_param_spec` | JSON | Ground truth parameters from dataset |
| `stl_path` | VARCHAR(512) | Path to ground truth STL file |
| `created_at` | DATETIME | Record creation timestamp |

**Example:**
```sql
INSERT INTO samples (id, category, text_desc, gt_param_spec, stl_path)
VALUES (
  'flange_00001',
  'Flange',
  'A circular flange with diameter 120mm...',
  '{"base_diameter": 120, "base_height": 15, "outer_diameter": 80, "inner_diameter": 40}',
  'data/LLM4CAD/Flange/mesh/flange_00001.stl'
);
```

---

### 2. `runs` - Agent Execution Runs

Tracks each invocation of the design agent on a sample.

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR(64) | Primary key (UUID) |
| `sample_id` | VARCHAR(64) | Foreign key → samples.id |
| `final_param_score` | FLOAT | Final parameter accuracy score (0-1) |
| `final_vlm_score` | FLOAT | Final VLM visual score (0-1) |
| `total_iterations` | INTEGER | Number of refinement iterations |
| `success` | BOOLEAN | Whether run met termination thresholds |
| `final_pred_param_spec` | JSON | Final predicted parameters |
| `final_cad_file` | VARCHAR(512) | Path to final STEP file |
| `final_render_image` | VARCHAR(512) | Path to final PNG render |
| `started_at` | DATETIME | Run start timestamp |
| `completed_at` | DATETIME | Run completion timestamp |
| `duration_seconds` | FLOAT | Total run duration |

**Example:**
```sql
INSERT INTO runs (id, sample_id, final_param_score, final_vlm_score, total_iterations, success)
VALUES (
  'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
  'flange_00001',
  0.95,
  0.88,
  2,
  TRUE
);
```

---

### 3. `nodes` - Node Execution Log

Records each LangGraph node execution within a run.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `run_id` | VARCHAR(64) | Foreign key → runs.id |
| `node_name` | VARCHAR(64) | Node identifier (e.g., "generate_param_spec") |
| `iteration` | INTEGER | Which iteration (0-indexed) |
| `input_state` | JSON | State passed into the node |
| `output_delta` | JSON | Changes produced by the node |
| `started_at` | DATETIME | Execution start |
| `completed_at` | DATETIME | Execution end |
| `duration_ms` | FLOAT | Execution time in milliseconds |
| `success` | BOOLEAN | Whether node executed successfully |
| `error_message` | TEXT | Error details if failed |

**Node Names:**
- `generate_param_spec` - LLM extracts parameters from text
- `judge_param_spec` - Compare predicted vs ground truth
- `generate_cad` - CadQuery generates STEP/STL
- `judge_cad_vlm` - VLM evaluates rendered CAD
- `decide_next_step` - Termination logic

**Example:**
```sql
INSERT INTO nodes (run_id, node_name, iteration, input_state, output_delta, duration_ms, success)
VALUES (
  'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
  'generate_param_spec',
  0,
  '{"text_desc": "A circular flange..."}',
  '{"pred_param_spec": {"base_diameter": 120, ...}}',
  1523.5,
  TRUE
);
```

---

### 4. `artifacts` - Generated Files

Tracks all files produced during runs.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `run_id` | VARCHAR(64) | Foreign key → runs.id |
| `artifact_type` | VARCHAR(32) | Type: "cad_step", "cad_stl", "render_png" |
| `file_path` | VARCHAR(512) | Full path to artifact file |
| `iteration` | INTEGER | Which iteration produced this |
| `file_size_bytes` | INTEGER | File size in bytes |
| `created_at` | DATETIME | Creation timestamp |

**Artifact Types:**
- `cad_step` - STEP format CAD file
- `cad_stl` - STL mesh file
- `render_png` - Rendered PNG image

**Example:**
```sql
INSERT INTO artifacts (run_id, artifact_type, file_path, iteration, file_size_bytes)
VALUES (
  'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
  'cad_step',
  'artifacts/a1b2c3d4/cad/flange_00001_iter0.step',
  0,
  45678
);
```

---

## Querying the Database

### Using Python (SQLAlchemy)

```python
from src.storage.database import get_database

db = get_database()

# Get a sample
sample = db.get_sample("flange_00001")

# Get all runs for a sample
runs = db.get_runs_for_sample("flange_00001")

# Get node executions for a run
nodes = db.get_nodes_for_run(run_id)

# Get statistics
stats = db.get_stats()
# Returns: {"total_samples": 100, "total_runs": 150, "successful_runs": 120, ...}
```

### Using SQLite CLI

```bash
# Open database
sqlite3 db/process.sqlite

# Show tables
.tables

# Show schema
.schema samples
.schema runs

# Query samples
SELECT id, category, json_extract(gt_param_spec, '$.base_diameter') as diameter 
FROM samples 
WHERE category = 'Flange' 
LIMIT 5;

# Query successful runs with scores
SELECT r.id, s.category, r.final_param_score, r.final_vlm_score, r.total_iterations
FROM runs r
JOIN samples s ON r.sample_id = s.id
WHERE r.success = 1
ORDER BY r.final_param_score DESC;

# Query node execution times
SELECT node_name, AVG(duration_ms) as avg_ms, COUNT(*) as count
FROM nodes
GROUP BY node_name
ORDER BY avg_ms DESC;

# Find artifacts for a run
SELECT artifact_type, file_path, file_size_bytes
FROM artifacts
WHERE run_id = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';
```

---

## File Storage Layout

Generated artifacts are stored alongside the database:

```
artifacts/
├── {run_id}/
│   ├── cad/
│   │   ├── flange_00001_iter0.step
│   │   └── flange_00001_iter1.step
│   └── renders/
│       ├── flange_00001_iter0.png
│       └── flange_00001_iter1.png
└── demo/
    ├── cad/
    └── renders/
```

---

## Maintenance

### Backup

```bash
# Create backup
cp db/process.sqlite db/process_backup_$(date +%Y%m%d).sqlite

# Or using sqlite3
sqlite3 db/process.sqlite ".backup db/backup.sqlite"
```

### Reset Database

```bash
# Delete and recreate
rm db/process.sqlite

# The database will be recreated on next pipeline run
```

### Vacuum (reclaim space)

```bash
sqlite3 db/process.sqlite "VACUUM;"
```

