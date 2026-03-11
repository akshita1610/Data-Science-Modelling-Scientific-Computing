# Contributing to Detector Calibration Simulation

Thank you for your interest in contributing to this project! This document provides guidelines for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git installed and configured
- Basic understanding of Python and scientific computing

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/detector-calibration-simulation.git
   cd detector-calibration-simulation
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   ```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Include type hints where appropriate

### Code Structure
- Keep modules focused and cohesive
- Follow the existing project structure
- Use the established patterns for new components

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Test with different Python versions (3.8, 3.9, 3.10, 3.11)

## 🤝 How to Contribute

### Reporting Issues

1. **Bug Reports**
   - Use the issue template
   - Include steps to reproduce
   - Provide system information (OS, Python version)
   - Include error messages and traceback

2. **Feature Requests**
   - Describe the use case
   - Explain why it would be valuable
   - Suggest implementation approach if possible

### Submitting Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new calibration method"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   - Create a pull request on GitHub
   - Fill out the PR template
   - Wait for code review

## 📋 Contribution Areas

### High Priority
- **Bug fixes** and performance improvements
- **Documentation** improvements
- **Test coverage** expansion

### New Features
- **Additional noise models** (e.g., 1/f noise, cosmic ray hits)
- **New signal sources** (e.g., line sources, pattern generators)
- **Advanced calibration** methods (e.g., machine learning-based)
- **Interactive features** (e.g., real-time parameter adjustment)
- **Export formats** (e.g., DICOM, FITS for astronomy)

### Examples and Documentation
- **Tutorial notebooks** for different use cases
- **Research paper** implementations
- **Case studies** and applications
- **API documentation** improvements

## 🔍 Code Review Process

### What We Look For
- **Correctness**: Does the code work as intended?
- **Style**: Is it readable and follows conventions?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well-documented?
- **Performance**: Does it maintain or improve performance?

### Review Guidelines
- Be constructive and respectful
- Explain issues clearly
- Suggest improvements
- Help new contributors learn

## 📖 Documentation

### Types of Documentation
- **Code comments**: Explain complex algorithms
- **Docstrings**: Document public APIs
- **README**: Update for new features
- **Examples**: Provide usage examples

### Documentation Standards
- Use clear, concise language
- Include code examples
- Explain mathematical concepts
- Provide troubleshooting tips

## 🧪 Testing

### Test Structure
```
tests/
├── test_models/           # Test detector and noise models
├── test_simulation/       # Test signal generation
├── test_calibration/      # Test calibration algorithms
├── test_visualization/    # Test plotting functions
├── test_evaluation/      # Test metrics and analysis
└── test_integration/     # End-to-end tests
```

### Test Guidelines
- Use descriptive test names
- Test edge cases and error conditions
- Use fixtures for common test data
- Aim for high code coverage

## 🏷️ Version Control

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

**Examples:**
```
feat(calibration): add wavelet denoising method

fix(models): resolve numpy compatibility issue

docs(readme): update installation instructions
```

## 🌟 Recognition

### Contributors
- All contributors are acknowledged in the README
- Significant contributions may be added as authors
- Feature requests and bug reports are valued

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive collaboration
- Report issues to maintainers

## 📞 Getting Help

### Resources
- **Documentation**: Check the README and docstrings
- **Issues**: Look for similar existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for private questions

### When Stuck
1. Check existing documentation and issues
2. Create a discussion with your question
3. Start with a simple implementation and ask for feedback
4. Join community channels (if available)

## 🚀 Release Process

### Version Management
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version numbers in `__init__.py`
- Create release notes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version numbers updated
- [ ] Changelog updated
- [ ] Tag created
- [ ] Release published

Thank you for contributing to the Detector Calibration Simulation! 🎯

---

For questions about contributing, please open an issue or start a discussion on GitHub.
