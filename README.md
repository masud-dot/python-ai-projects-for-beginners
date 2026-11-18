# Python AI Projects for Beginners

**10 End-to-End Projects from Classic ML to Modern Generative AI**

A practical companion repository containing the code examples and projects from the book *Python AI Projects for Beginners: 10 End-to-End Projects from Classic ML to Modern Generative AI*.

This repository is designed to help beginners learn AI and machine learning through hands-on projects, progressing from foundational Python concepts to modern generative AI applications.

---

## About This Repository

This repository contains the project code organized by chapter. Each chapter folder includes the relevant Python scripts, supporting files, and tests where applicable.

The goal is to make it easy to:

- Follow the projects while reading the book.
- Run the examples independently.
- Experiment with the code and modify it.
- Understand how different AI and machine learning techniques work in practice.
- Build a foundation for developing more advanced AI applications.

**Important:** Use Python 3.12 for the repository setup unless a particular chapter specifies additional requirements.

## What You'll Build

The book covers 10 practical projects, progressing from traditional machine learning to generative AI.

| Chapter | Project | What You'll Explore |
|---|---|---|
| 01 | Environment Setup | Python environment and project setup |
| 02 | Search and Optimization | Search algorithms and optimization techniques |
| 03 | House Prices | A practical machine learning regression project |
| 04 | Handwritten Digits | Image classification fundamentals |
| 05 | NumPy Neural Network | Building a neural network using NumPy |
| 06 | PyTorch | Deep learning with PyTorch |
| 07 | Transfer Learning | Reusing pretrained models |
| 08 | Whisper | Speech recognition with Whisper |
| 09 | Hugging Face Sentiment Analysis | Natural language processing and sentiment classification |
| 10 | Local AI Chatbot | Running a chatbot locally |
| 11 | Local RAG | Building a local Retrieval-Augmented Generation application |
| 13 | Appendix A — Worked Answers | Solutions to the book's practice exercises |

**Why does the repository include `ch13`?**

The `ch13` folder contains the worked answers from Appendix A. It is intentionally numbered to match the source material's organization and is not a missing project chapter.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── .devcontainer/
│
├── ch01/
├── ch02/
├── ch03/
├── ch04/
├── ch05/
├── ch06/
├── ch07/
├── ch08/
├── ch09/
├── ch10/
├── ch11/
│
├── ch13/
│
├── README.md
└── LICENSE
```

Each chapter is maintained in its own directory to keep the projects independent and easy to navigate.

Individual chapter folders may contain their own `requirements.txt` files. Install only the dependencies needed for the project you intend to run.

---

## Prerequisites

Before getting started, make sure you have:

- Python 3.12
- pip (Python package installer)
- Git
- A code editor such as Visual Studio Code or PyCharm

Some projects require additional software or resources. For example, the local chatbot and RAG projects may require Ollama and a locally available language model.

Check the relevant chapter's instructions before running a project.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Replace the URL with the actual repository URL.

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can activate the environment using Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Project Dependencies

Navigate to the chapter you want to explore.

For example:

```bash
cd ch03
python -m pip install -r requirements.txt
```

Not every chapter necessarily requires a separate dependency installation. Check the relevant folder before installing packages.

---

## Running the Projects

Each chapter is designed to be explored independently.

For example, to run the House Prices project:

```bash
cd ch03
python <script-name>.py
```

Replace `<script-name>.py` with the actual Python script in that chapter.

For other projects, navigate to the relevant chapter folder and follow its instructions.

**Tip:** Read the chapter's code and documentation before running it. Some projects may require datasets, model downloads, local services, or additional configuration.

---

## Running Tests

The repository includes tests for the chapter projects.

Make sure you have installed the dependencies required by the chapter you want to test.

### Run Tests for One Chapter

From the repository root:

```bash
python -m pytest ch03
```

Replace `ch03` with the chapter you want to test.

Alternatively, navigate into a chapter folder:

```bash
cd ch03
python -m pytest
```

### Run All Available Tests

From the repository root:

```bash
python -m pytest
```

### Run Tests with Detailed Output

```bash
python -m pytest -v
```

Test availability and execution requirements may differ between chapters.

---

## GitHub Actions

This repository includes a GitHub Actions workflow at:

```text
.github/workflows/tests.yml
```

The workflow is intended to run the repository's automated tests when changes are pushed or pull requests are opened.

Check the **Actions** tab on GitHub to see the latest workflow results.

---

## GitHub Codespaces

The repository includes a `.devcontainer/` configuration intended to support GitHub Codespaces.

If the configuration is maintained and available, you can use Codespaces to work with the projects in a browser-based development environment without setting up everything locally.

To get started:

1. Open the repository on GitHub.
2. Select the **Code** button.
3. Open the **Codespaces** tab.
4. Create a new Codespace.

---

## Troubleshooting

### Python Version Issues

Verify your Python version:

```bash
python --version
```

Use Python 3.12 for the repository's recommended environment.

### Package Installation Problems

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then install the requirements for the relevant chapter again.

### Virtual Environment Issues

Make sure your virtual environment is activated before installing dependencies or running tests.

### Missing Dependencies

Navigate to the relevant chapter and install its requirements:

```bash
python -m pip install -r requirements.txt
```

### Test Failures

Run the tests with verbose output:

```bash
python -m pytest -v
```

Review the error message and verify that the required dependencies, datasets, model files, and external services are available.

---

## Contributing

Contributions, suggestions, and corrections are welcome.

If you find an issue or want to improve a project:

1. Fork this repository.
2. Create a new branch.
3. Make your changes.
4. Run the relevant tests.
5. Submit a pull request with a clear description.

For bugs or questions, you can also open a GitHub Issue.

Please keep contributions consistent with the book's educational approach and beginner-friendly explanations.

---

## License

This repository is distributed under the **MIT License**.

See the [LICENSE](LICENSE) file for the complete license text.

---

## About the Book

**Python AI Projects for Beginners: 10 End-to-End Projects from Classic ML to Modern Generative AI**

Learn artificial intelligence through practical Python projects, starting with traditional machine learning and progressing to deep learning, natural language processing, local AI chatbots, and Retrieval-Augmented Generation.

This repository serves as the companion code resource for the book.

**Book:** [Add your Amazon book link here]

**Companion Repository:** (https://github.com/masud-dot/python-ai-projects-for-beginners)

---

*Learn by building. Experiment by doing. Keep exploring AI.*
