## 1) Install Python 3.12

**Windows (PowerShell)**

```powershell
winget install -e --id Python.Python.3.12
python --version
```

## 2) Create a virtual env & install deps

```bash
python3.12 -m venv .venv
source .venv/bin/activate 
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**requirements.txt (pinned)**

```txt
ragas==0.3.4
pandas==2.3.2
python-dotenv==1.1.1
tqdm==4.67.1
rich==14.1.0
openai==1.107.3
langchain-openai==0.3.33
datasets==4.0.0
pyarrow==17.0.0
```

---

## 3) Add your OpenAI key

Create `.env` in the project root:

```dotenv
OPENAI_API_KEY=sk-YOUR_KEY_HERE
OPENAI_MODEL=gpt-4o


## 4) Run the app

```bash
python run_ragas_pizza.py
pytest -q
```
