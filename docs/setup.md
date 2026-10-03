# Setup

## Google Colab

1. Open a notebook directly from GitHub, for example
   `https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/04_gru.ipynb`
   (or in Colab: *File → Open notebook → GitHub*).
2. Choose a CPU or GPU runtime (*Runtime → Change runtime type*). A T4 GPU trains the recurrent models faster, but every logged run except the LSTM's was made on CPU, and a GPU can change the last digits of a recurrent model's results.
3. Run the setup cell below. It is the first code cell of every notebook.
4. When asked, upload `Train.csv`, or load it from Drive (see [data.md](data.md)).

```python
# Setup: works on Google Colab and locally.
import subprocess
import sys
from pathlib import Path

REPO = "vaccine-sentiment-sequential-models"
if "google.colab" in sys.modules:
    ROOT = Path("/content") / REPO
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/Adossi-design/{REPO}.git", str(ROOT)], check=True)
    else:  # get teammates' latest pushes (e.g. the split files)
        subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    str(ROOT / "requirements.txt")], check=True)
else:
    ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "common.py").exists())
sys.path.insert(0, str(ROOT))

from src import common

common.print_environment()
common.set_seed()
```

The cell clones `main` (or pulls the latest `main` if the clone already exists), installs `requirements.txt` (a no-op on Colab, whose preinstalled versions already satisfy it) and imports the shared helpers.

Files a notebook writes on Colab (split IDs, predictions, experiment logs, figures, models) disappear when the session ends. Download them and commit them from a local clone. Notebooks 02 to 05 need `splits/` on `main` before they can run.

Notebook 04's round 4 starts the GRU's embedding from GloVe Twitter vectors. It reads the vectors for the GRU's vocabulary from `models/gru_glove_100d.npz`, which the clone includes, and downloads the 1.5 GB GloVe archive into `data/` only if that file is missing or no longer matches the vocabulary (see [data/README.md](../data/README.md)).

## Local

Use Python 3.10 to 3.13 (Colab uses 3.13). TensorFlow 2.21 has no wheels for Python 3.14.

```bash
git clone https://github.com/Adossi-design/vaccine-sentiment-sequential-models.git
cd vaccine-sentiment-sequential-models
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab
```

Put `Train.csv` in `data/` first.
