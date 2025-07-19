#!/bin/bash
# run_all_tests.sh - Run all tests for the speak-app project
# This script runs the complete test suite with various options

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
VERBOSE=""
COVERAGE=""
MARKERS=""
SPECIFIC_FILE=""
FAILED_FIRST=""

# Function to display help
show_help() {
    cat << EOF
run_all_tests.sh - Run speak-app test suite

USAGE:
    ./run_all_tests.sh [OPTIONS]

OPTIONS:
    -v, --verbose       Run tests in verbose mode
    -c, --coverage      Run with coverage report
    -m, --marker MARKER Run only tests with specific marker (unit, integration, api, etc.)
    -f, --file FILE     Run specific test file
    -x, --failed-first  Run failed tests first
    -h, --help          Show this help message

MARKERS:
    unit         Unit tests
    integration  Integration tests
    api          API tests
    elevenlabs   ElevenLabs-specific tests
    voice        Voice-related tests
    regression   Regression tests
    slow         Slow tests

EXAMPLES:
    ./run_all_tests.sh                    # Run all tests
    ./run_all_tests.sh -v                 # Run with verbose output
    ./run_all_tests.sh -c                 # Run with coverage
    ./run_all_tests.sh -m unit           # Run only unit tests
    ./run_all_tests.sh -f test_observability.py  # Run specific file
    ./run_all_tests.sh -x                 # Run failed tests first

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=tts --cov-report=term-missing"
            shift
            ;;
        -m|--marker)
            MARKERS="-m $2"
            shift 2
            ;;
        -f|--file)
            SPECIFIC_FILE="tests/$2"
            shift 2
            ;;
        -x|--failed-first)
            FAILED_FIRST="--failed-first"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to run tests
run_tests() {
    local test_command="pytest"
    
    # Add options
    [ -n "$VERBOSE" ] && test_command="$test_command $VERBOSE"
    [ -n "$COVERAGE" ] && test_command="$test_command $COVERAGE"
    [ -n "$MARKERS" ] && test_command="$test_command $MARKERS"
    [ -n "$FAILED_FIRST" ] && test_command="$test_command $FAILED_FIRST"
    
    # Add color output
    test_command="$test_command --color=yes"
    
    # Add test file or directory
    if [ -n "$SPECIFIC_FILE" ]; then
        test_command="$test_command $SPECIFIC_FILE"
    else
        test_command="$test_command tests/"
    fi
    
    # Add summary options
    test_command="$test_command --tb=short --durations=10"
    
    echo -e "${BLUE}Running command: $test_command${NC}"
    echo ""
    
    # Run the tests
    eval $test_command
}

# Main execution
echo -e "${GREEN}=== Speak App Test Suite ===${NC}"
echo ""

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Please install pytest: pip install pytest pytest-mock pytest-cov"
    exit 1
fi

# Run the tests
run_tests

# Exit with the test result code
exit $?