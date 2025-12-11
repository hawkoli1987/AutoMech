#!/bin/bash
# Run all tests for the Agentic CAD Framework
# Usage: ./scripts/run_tests.sh [options]
#
# Options:
#   -l, --live    Include live LLM/VLM tests (requires running servers)
#   -v            Verbose output
#   -h, --help    Show this help

set -e

cd "$(dirname "$0")/.."

# Parse arguments
LIVE_TESTS=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--live)
            LIVE_TESTS="1"
            shift
            ;;
        -v)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            head -15 "$0" | tail -12
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "Agentic CAD Framework - Test Suite"
echo "================================================"
echo ""

# Set environment for tests
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:8001}
export OPENAI_API_BASE2=${OPENAI_API_BASE2:-http://localhost:8002}

echo "LLM API: $OPENAI_API_BASE"
echo "VLM API: $OPENAI_API_BASE2"
echo ""

if [ -z "$LIVE_TESTS" ]; then
    echo "Running unit tests (excluding live tests)..."
    echo ""
    python3 -m pytest tests/ $VERBOSE --ignore=tests/test_llm_client.py -k "not Live" --tb=short
    
    echo ""
    echo "Running LLM client tests (mocked)..."
    python3 -m pytest tests/test_llm_client.py $VERBOSE -k "not Live" --tb=short
else
    echo "Running ALL tests including live LLM/VLM tests..."
    echo ""
    python3 -m pytest tests/ $VERBOSE --tb=short
fi

echo ""
echo "================================================"
echo "All tests completed!"
echo "================================================"

