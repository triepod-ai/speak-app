#!/bin/bash
# run_audio_tests.sh - Run manual audio tests for the speak-app project
# These tests produce actual audio output for verification

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
TEST_TYPE="all"
INTERACTIVE=false
SPECIFIC_TEST=""
VERBOSE=""

# Function to display help
show_help() {
    cat << EOF
run_audio_tests.sh - Run manual audio tests for speak-app

USAGE:
    ./run_audio_tests.sh [OPTIONS]

OPTIONS:
    -t, --type TYPE        Test type to run (all, basic, elevenlabs, providers, comparison)
    -i, --interactive      Run interactive tests
    -s, --specific TEST    Run specific test function
    -v, --verbose          Verbose output
    -h, --help             Show this help message

TEST TYPES:
    all         Run all audio tests (default)
    basic       Run basic audio tests only
    elevenlabs  Run ElevenLabs voice tests
    providers   Run provider comparison tests
    comparison  Run side-by-side comparisons

EXAMPLES:
    ./run_audio_tests.sh                          # Run all tests
    ./run_audio_tests.sh -t elevenlabs           # Run ElevenLabs tests only
    ./run_audio_tests.sh -t basic -i             # Run basic tests with interactive mode
    ./run_audio_tests.sh -s test_rachel_voice    # Run specific test

NOTES:
    - These tests produce actual audio output
    - Make sure your speakers/headphones are connected
    - API keys must be set for cloud providers
    - Use Ctrl+C to stop tests at any time

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -s|--specific)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
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

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        echo -e "${RED}Error: pytest is not installed${NC}"
        echo "Please install pytest: pip install pytest pytest-mock"
        exit 1
    fi
    
    # Check for API keys
    echo -e "\n${CYAN}API Key Status:${NC}"
    if [ -n "${ELEVENLABS_API_KEY:-}" ]; then
        echo -e "  ${GREEN}✓${NC} ElevenLabs API key found"
    else
        echo -e "  ${YELLOW}⚠${NC}  ElevenLabs API key not found (some tests will be skipped)"
    fi
    
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        echo -e "  ${GREEN}✓${NC} OpenAI API key found"
    else
        echo -e "  ${YELLOW}⚠${NC}  OpenAI API key not found (some tests will be skipped)"
    fi
    
    echo -e "  ${GREEN}✓${NC} pyttsx3 (offline) is always available"
    echo ""
}

# Function to run tests
run_tests() {
    local test_file=""
    local test_args="-v -s -m audio_output"
    
    # Add verbose flag if requested
    if [ -n "$VERBOSE" ]; then
        test_args="$test_args $VERBOSE"
    fi
    
    # Add interactive marker if requested
    if [ "$INTERACTIVE" = true ]; then
        test_args="$test_args -m interactive"
    fi
    
    # Determine which test file to run
    case $TEST_TYPE in
        all)
            echo -e "${GREEN}Running all audio tests...${NC}"
            test_file="tests/test_manual_*.py"
            ;;
        basic)
            echo -e "${GREEN}Running basic audio tests...${NC}"
            test_file="tests/test_manual_audio.py"
            ;;
        elevenlabs)
            echo -e "${GREEN}Running ElevenLabs voice tests...${NC}"
            test_file="tests/test_manual_elevenlabs_voices.py"
            ;;
        providers|comparison)
            echo -e "${GREEN}Running provider comparison tests...${NC}"
            test_file="tests/test_manual_provider_comparison.py"
            ;;
        *)
            echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
            exit 1
            ;;
    esac
    
    # Add specific test if requested
    if [ -n "$SPECIFIC_TEST" ]; then
        test_args="$test_args -k $SPECIFIC_TEST"
        echo -e "${CYAN}Running specific test: $SPECIFIC_TEST${NC}"
    fi
    
    # Show test command
    echo -e "${BLUE}Running command: pytest $test_file $test_args${NC}"
    echo ""
    
    # Run the tests
    pytest $test_file $test_args
}

# Function to show audio test menu
show_menu() {
    echo -e "${CYAN}=== Speak App Audio Test Menu ===${NC}"
    echo ""
    echo "1. Run all audio tests"
    echo "2. Test basic TTS functionality"
    echo "3. Test ElevenLabs voices (Rachel, Domi, Bella, Adam)"
    echo "4. Compare all providers side-by-side"
    echo "5. Run interactive tests"
    echo "6. Run specific test by name"
    echo "7. Show available tests"
    echo "8. Exit"
    echo ""
    read -p "Select option (1-8): " choice
    
    case $choice in
        1)
            TEST_TYPE="all"
            run_tests
            ;;
        2)
            TEST_TYPE="basic"
            run_tests
            ;;
        3)
            TEST_TYPE="elevenlabs"
            run_tests
            ;;
        4)
            TEST_TYPE="comparison"
            run_tests
            ;;
        5)
            INTERACTIVE=true
            TEST_TYPE="all"
            run_tests
            ;;
        6)
            read -p "Enter test name (e.g., test_rachel_voice): " SPECIFIC_TEST
            TEST_TYPE="all"
            run_tests
            ;;
        7)
            echo -e "\n${CYAN}Available test files:${NC}"
            ls -la tests/test_manual_*.py | awk '{print "  " $9}'
            echo -e "\n${CYAN}To see specific tests in a file:${NC}"
            echo "  pytest tests/test_manual_audio.py --collect-only"
            echo ""
            read -p "Press Enter to continue..."
            show_menu
            ;;
        8)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            show_menu
            ;;
    esac
}

# Main execution
main() {
    echo -e "${GREEN}=== Speak App Audio Test Runner ===${NC}"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Show warning
    echo -e "${YELLOW}⚠️  WARNING: These tests will produce audio output!${NC}"
    echo -e "${YELLOW}   Make sure your speakers/headphones are at a comfortable volume.${NC}"
    echo ""
    
    # If no arguments, show menu
    if [ "$TEST_TYPE" = "all" ] && [ "$INTERACTIVE" = false ] && [ -z "$SPECIFIC_TEST" ]; then
        show_menu
    else
        run_tests
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Tests interrupted by user${NC}"; exit 1' INT

# Run main function
main

# Exit with test result code
exit $?