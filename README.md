# Python AI Projects for Beginners: 10 End-to-End Projects from Classic ML to Modern Generative AI: the code

The complete, tested code for every project in *Python AI Projects for Beginners: 10 End-to-End Projects from Classic ML to Modern Generative AI*. Each folder holds one chapter's code, exactly as the book prints it.

| Folder | Chapter | Python files | Tests |
|---|---|---|---|
| [`ch01/`](ch01/) | Set Up Your Machine and Your Project Folder | 6 | yes |
| [`ch02/`](ch02/) | Project 1: Search and Optimization in Python | 10 | yes |
| [`ch03/`](ch03/) | Project 2: Predict House Prices with Scikit-Learn | 13 | yes |
| [`ch04/`](ch04/) | Project 3: Recognize Handwritten Digits and Study the Mistakes | 11 | yes |
| [`ch05/`](ch05/) | Project 4: Build a Neural Network from Scratch in NumPy | 18 | yes |
| [`ch06/`](ch06/) | Project 5: PyTorch Fundamentals: Tensors, Autograd and a Training Loop | 13 | yes |
| [`ch07/`](ch07/) | Project 6: Classify Your Own Photos with Transfer Learning | 8 | yes |
| [`ch08/`](ch08/) | Project 7: Transcribe Audio to Text with Whisper | 7 | yes |
| [`ch09/`](ch09/) | Project 8: Analyze Text Sentiment with Hugging Face Pipelines | 9 | yes |
| [`ch10/`](ch10/) | Project 9: Build a Chatbot That Runs on Your Own Computer | 10 | yes |
| [`ch11/`](ch11/) | Project 10: Ask Questions About Your Own Documents with Local RAG | 13 | yes |
| [`ch13/`](ch13/) | Appendix A: Worked Answers to the Extend the Project Exercises | 16 | yes |

## Set up

You need Python 3.14. Then, in this folder:

```bash
python3.14 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Each chapter folder also has its own `requirements.txt` when it needs one, so you can install only what a single project uses.

## Run the tests

```bash
python -m pytest ch01
```

Change `ch01` to the chapter you are working on, or run `python -m pytest` inside a chapter folder.

## Put it on GitHub

1. Create an empty repository on GitHub (no README, no licence).
2. In this folder run:

```bash
git init
git add .
git commit -m "Book code"
git branch -M main
git remote add origin https://github.com/<you>/<repository>.git
git push -u origin main
```

GitHub Actions then runs every chapter's tests (`.github/workflows/tests.yml`), and **Code > Codespaces > Create codespace** opens the code in a ready-to-run environment (`.devcontainer/`).

## Licence

[LICENCE: the author adds the licence before publishing]
