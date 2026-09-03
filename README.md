# MLA-experiments — Reproduktionscode zur Seminararbeit über Multi-Head Latent Attention

Dieses Repository enthält den vollständigen Code, die Konfigurationen und die Rohdaten der vier
Trainingsstudien zur Seminararbeit über **Multi-Head Latent Attention (MLA, DeepSeek-V2)**.

Es ist bewusst reduziert: keine Ergebnisberichte, keine Abbildungen, keine Texte der Seminararbeit.
Enthalten ist nur, was zum Nachstellen der Experimente und zum Nachrechnen der berichteten
Statistiken nötig ist. Ausarbeitung, Ergebnisberichte, Abbildungen und Literatur stehen im
Hauptrepository: <https://github.com/TimAlexanderStrauss/SeminarML>.

## Die vier Studien

| Ordner | Design | Runs |
|---|---|---|
| `experiments/` | 2×2-Ablation: Low-Rank KV × Decoupled RoPE, Dense-Backbone | 12 |
| `experiments/` (Sweep) | $d_c$-Kompressions-Sweep über 4 Kompressionsstufen × 2 Seeds | 8 |
| `experiments/paper_layout_followup/` | Layout-Sensitivität: MHA vs. bisheriges MLA vs. DeepSeek-näheres MLA, 3 Seeds | 9 |
| `experiments/moe_followup/` | Dieselbe 2×2-Attention-Ablation im DeepSeekMoE-Backbone, 3 Seeds | 12 |
| `experiments/deepseek_mechanism_2x2x2/` | Gemeinsames 2×2×2: Low-Rank KV × Decoupled RoPE × Dense/MoE, 3 Seeds | 24 |

Insgesamt 65 Trainingsruns, rund 155 GPU-Stunden auf einer RTX 5070.

Jede Studie ist eigenständig: eigenes `model.py`, `train.py`, eigene Runner, eigene Tests und eine
eigene Auswertung. Sie teilen sich nur den Datensatz und die `requirements.txt`.

## Was hier liegt — und was nicht

**Enthalten**

- `model.py` / `train.py` je Studie — Architektur und Trainingsschleife
- `run_experiments.*`, `run_sweep.ps1`, `run_smoke_test.ps1` — Runner mit den exakten Hyperparametern
- `test_correctness.py`, `test_reference.py`, `test_gpu_capacity.py` — Korrektheits- und Referenztests
- `analyze_results.py`, `analyze_sweep.py`, `analyze.ps1` — die Auswertung, die aus den Rohdaten die
  berichteten Kontraste, Tests und Plots erzeugt
- `data/fineweb_edu/prepare.py` — Download und Tokenisierung der Trainingsdaten
- `results*/` — die Rohdaten aller 65 abgeschlossenen Runs: `config.json` (exakte Hyperparameter des
  Runs), `metrics.csv` (Loss-Verlauf), teilweise `routing.csv` (Router-Lasten) und `runtime.json`
  (Durchsatz). Damit lässt sich die Auswertung ohne erneutes Training nachrechnen und eine eigene
  Reproduktion gegen die Originalzahlen prüfen.
- `SETUP_DESKTOP.md`, die Studien-READMEs und `RTX5070_CHECKLIST.md` — Setup, Versuchsdesign und
  Runbook

**Nicht enthalten** (steht im Hauptrepository)

- Ergebnisberichte und Diskussion (`RESULTS.md`, `SWEEP_RESULTS.md`, `EXPERIMENT_UEBERBLICK.md`)
- Abbildungen und deren Erläuterung (`plots/`, `figures/`, `ABBILDUNGEN.md`)
- Reviews, Text der Seminararbeit, Präsentation, Literaturverzeichnis

Die `plots/`-Ordner fehlen absichtlich — die Auswertungsskripte legen sie beim Ausführen neu an.

## Reproduktion

### 1. Umgebung

Python 3.11 oder 3.12. PyTorch ≥ 2.6 mit passendem CUDA-Build; für Blackwell (RTX 5070, sm_120) ist
CUDA 12.8 nötig.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
cd experiments
pip install -r requirements.txt
```

### 2. Daten

FineWeb-Edu, 500M Trainings- und 5M Validierungstoken, GPT-2-BPE, deterministischer Split. Ein
HuggingFace-Account mit akzeptierten Nutzungsbedingungen für
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) ist nötig.

```bash
cd experiments
python data/fineweb_edu/prepare.py   # ~15–30 min, ~1 GB
```

`train.bin` und `val.bin` werden nicht versioniert (siehe `.gitignore`) und müssen einmalig erzeugt
werden. Alle vier Studien greifen auf dieselben Dateien zu.

### 3. Tests vor dem Training

```bash
cd experiments
python test_correctness.py
python test_reference.py
```

Die Folgestudien haben eigene Tests (`run_tests.ps1` bzw. direkt `python test_correctness.py` im
jeweiligen Ordner).

### 4. Training

Hauptversuch (2×2-Ablation, 12 Runs, ~21 h):

```bash
cd experiments
bash run_experiments.sh              # oder: .\run_experiments.ps1
bash run_experiments.sh mha          # nur eine Bedingung
bash run_experiments.sh mla 42       # nur ein Run
```

$d_c$-Sweep (8 Runs):

```powershell
.\run_sweep.ps1
```

Folgestudien:

```powershell
cd experiments\paper_layout_followup
.\run_experiments.ps1

cd ..\moe_followup
.\run_tests.ps1 ; .\run_smoke_test.ps1 ; .\run_experiments.ps1

cd ..\deepseek_mechanism_2x2x2
.\run_tests.ps1 ; .\run_smoke_test.ps1
python benchmark_gpu.py              # erzeugt gpu_profile.json (GPU-Autotuning)
.\run_experiments.ps1                # oder: bash run_experiments.sh
```

Runs sind unterbrechbar: abgeschlossene Runs werden über die `DONE`-Flag-Dateien übersprungen,
laufende setzen vom letzten Checkpoint fort. Checkpoints (`*.pt`, ~400 MB pro Run) werden nicht
versioniert und können nach der Auswertung gelöscht werden.

### 5. Auswertung

Die Auswertung liest ausschließlich `results*/`. Sie läuft daher auch ohne eigenes Training direkt
auf den mitgelieferten Rohdaten — das ist der schnellste Weg, die berichteten Zahlen nachzurechnen.

```bash
cd experiments
python analyze_results.py            # 2×2: Lernkurven, Final-Loss, Heatmap, ANOVA
python analyze_sweep.py              # Sweep: Dose-Response, Pareto
```

```powershell
cd experiments\paper_layout_followup     ; .\analyze.ps1
cd ..\moe_followup                       ; .\analyze.ps1
cd ..\deepseek_mechanism_2x2x2           ; .\analyze.ps1
```

Ausgabe je Studie: `plots/` mit Abbildungen, CSV-Zusammenfassungen und einem automatisch erzeugten
`RESULTS_GENERATED.md`.

## Hardware und Laufzeit

Alle Runs entstanden auf einer RTX 5070 (12 GB, Blackwell). Richtwerte: 2×2-Ablation ~21 h,
Layout-Follow-up ~16 GPU-h, MoE-Follow-up ~45 GPU-h, 2×2×2 ~59 GPU-h. Details und
Speicheranforderungen in `experiments/SETUP_DESKTOP.md` und
`experiments/deepseek_mechanism_2x2x2/RTX5070_CHECKLIST.md`.

Auf anderer Hardware sind identische Zahlen nicht zu erwarten: cuDNN-Kernel, `torch.compile` und
TF32-Verhalten unterscheiden sich zwischen GPU-Generationen. Die Runs sind pro Seed RNG-gepaart, die
Kontraste sollten sich im Vorzeichen und in der Größenordnung reproduzieren.
