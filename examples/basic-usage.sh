#!/bin/bash
# basic-usage.sh - Basic examples of using the speak command

# Ensure speak is in PATH
if ! command -v speak &> /dev/null; then
    echo "Error: speak command not found in PATH"
    exit 1
fi

echo "=== Basic Speak Command Examples ==="
echo

# Example 1: Simple text
echo "1. Speaking simple text:"
speak "Hello, this is a test of the speak command"
sleep 2

# Example 2: Using variables
echo -e "\n2. Speaking with variables:"
USER_NAME="${ENGINEER_NAME:-User}"
speak "Welcome, $USER_NAME. The current time is $(date +%I:%M%p)"
sleep 2

# Example 3: Command output
echo -e "\n3. Speaking command output:"
FILE_COUNT=$(ls -1 | wc -l)
speak "There are $FILE_COUNT files in this directory"
sleep 2

# Example 4: Piped input
echo -e "\n4. Using piped input:"
echo "This text is piped to the speak command" | speak
sleep 2

# Example 5: Reading from file
echo -e "\n5. Reading file contents:"
echo "This is content from a temporary file" > /tmp/speak_test.txt
cat /tmp/speak_test.txt | speak
rm /tmp/speak_test.txt
sleep 2

# Example 6: Multiple providers
echo -e "\n6. Testing different providers:"
echo "   Available providers:"
speak --list

# Example 7: Status check
echo -e "\n7. Checking TTS status:"
speak --status

# Example 8: Silent mode
echo -e "\n8. Silent mode (no audio):"
speak --off "This message will not be spoken, only displayed"
echo "   (Message was not spoken due to --off flag)"

# Example 9: Math results
echo -e "\n9. Speaking calculation results:"
RESULT=$((42 * 2))
speak "The answer to 42 times 2 is $RESULT"
sleep 2

# Example 10: Conditional speech
echo -e "\n10. Conditional speech based on success/failure:"
if ping -c 1 google.com &> /dev/null; then
    speak "Internet connection is working"
else
    speak "No internet connection detected"
fi

echo -e "\n=== Basic examples complete ==="