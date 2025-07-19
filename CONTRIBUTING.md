# Contributing to Speak

Thank you for your interest in contributing to the Speak TTS command! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Adding New TTS Providers](#adding-new-tts-providers)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub: https://github.com/triepod-ai/speak-app
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Test thoroughly
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11+
- `uv` package manager
- Bash 4.0+
- Git

### Local Development

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/speak-app.git
cd speak-app

# Add upstream remote
git remote add upstream https://github.com/triepod-ai/speak-app.git

# Create a development branch
git checkout -b feature/your-feature-name

# Test the installation
./speak --test
```

## Adding New TTS Providers

To add a new TTS provider, follow these steps:

### 1. Create Provider Script

Create a new file in the `tts/` directory:

```python
# tts/yourprovider_tts.py
#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "your-required-package>=1.0.0",
# ]
# ///

"""
YourProvider TTS provider for speak command.
Brief description of the provider.
"""

import os
import sys

def speak_with_yourprovider(text: str) -> bool:
    """
    Convert text to speech using YourProvider.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Your implementation here
        # 1. Check for API key if needed
        # 2. Make API request or local synthesis
        # 3. Play audio
        return True
    except Exception as e:
        print(f"YourProvider TTS error: {e}", file=sys.stderr)
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: yourprovider_tts.py <text>", file=sys.stderr)
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    success = speak_with_yourprovider(text)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

### 2. Update Provider Selection

Modify `tts/tts_provider.py` to include your provider:

```python
# In get_available_providers()
if os.getenv("YOURPROVIDER_API_KEY"):
    providers.append("yourprovider")

# In script_map dictionary
"yourprovider": self.utils_dir / "yourprovider_tts.py",
```

### 3. Update Documentation

- Add provider to README.md provider comparison table
- Document required environment variables
- Add usage examples
- Update CHANGELOG.md

### 4. Test Your Provider

```bash
# Test directly
python3 tts/yourprovider_tts.py "Test message"

# Test through speak command
export YOURPROVIDER_API_KEY="your_key"
speak --provider yourprovider "Test message"
```

## Code Style

### Bash Style Guide

- Use `set -euo pipefail` for error handling
- Quote all variables: `"$VAR"`
- Use `[[ ]]` for conditionals
- Follow Google Shell Style Guide
- Add comments for complex logic

### Python Style Guide

- Follow PEP 8
- Use type hints where appropriate
- Document all functions with docstrings
- Handle exceptions gracefully
- Use f-strings for formatting

### General Guidelines

- Keep functions small and focused
- Use descriptive variable names
- Add error messages that help users
- Log errors to stderr, not stdout
- Maintain backward compatibility

## Testing

### Manual Testing

Test all major functionality:

```bash
# Basic functionality
speak "Test message"
echo "Piped test" | speak

# Provider selection
speak --list
speak --provider pyttsx3 "Offline test"

# Configuration
speak --status
speak --enable
speak --disable

# Error handling
speak --provider nonexistent "Should fail gracefully"
```

### Automated Testing

```bash
# Run test suite (if available)
./test/run_tests.sh

# Test specific provider
./test/test_provider.sh elevenlabs
```

### Test Coverage

Ensure your changes include tests for:
- Happy path functionality
- Error conditions
- Edge cases
- Provider fallback behavior

## Submitting Changes

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types: feat, fix, docs, style, refactor, test, chore

### Pull Request Process

1. Sync your fork with upstream:
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

2. Update documentation for your changes

3. Add tests for new functionality

4. Ensure all tests pass:
   ```bash
   ./run_all_tests.sh
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request on GitHub:
   - Go to https://github.com/triepod-ai/speak-app
   - Click "Pull requests" → "New pull request"
   - Select your fork and branch
   - Fill in the PR template with details about your changes
   - Submit the PR for review

7. Update CHANGELOG.md if needed

8. Respond to review feedback promptly

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tested locally
- [ ] Added new tests
- [ ] All tests pass

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Reporting Issues

Please report issues on GitHub: https://github.com/triepod-ai/speak-app/issues

### Bug Reports

Include:
- Speak version (`speak --version` when implemented)
- Operating system and version
- Python version
- Error messages
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests

Submit feature requests as GitHub issues with:
- Use case description
- Proposed implementation
- Alternative solutions considered
- Impact on existing functionality

### Security Issues

For security vulnerabilities:
- Do NOT open a public issue
- Email security concerns privately
- Include detailed description
- Suggest fixes if possible

## Development Tips

### Debugging

```bash
# Enable debug output
export DEBUG=1
speak "Test message"

# Test provider directly
python3 -m pdb tts/tts_provider.py "Debug test"
```

### Performance

- Minimize import time
- Use lazy loading where possible
- Cache API responses appropriately
- Profile slow operations

### Compatibility

- Test on multiple platforms (Linux, macOS, WSL)
- Support both old and new Python versions
- Handle missing dependencies gracefully
- Provide clear error messages

## Questions?

If you have questions about contributing:
1. Check existing issues and PRs
2. Read the documentation
3. Open a discussion issue
4. Contact maintainers

Thank you for contributing to Speak!