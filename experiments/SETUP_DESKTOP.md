# Setup auf Desktop-PC (RTX 5070)

Schritt-für-Schritt-Anleitung, damit die Experimente auf deinem Desktop laufen.

> **Hinweis zu diesem Repository:** Dies ist das Reproduktions-Repository — es enthält nur Code,
> Konfiguration und die Rohdaten der Runs. Die im Text erwähnten Ergebnisberichte (`RESULTS.md`,
> `SWEEP_RESULTS.md`, `ABBILDUNGEN.md`) und die fertigen Abbildungen liegen im Hauptrepository
> <https://github.com/TimAlexanderStrauss/SeminarML>. Die `plots/`-Ordner hier entstehen beim
> Ausführen der `analyze_*`-Skripte neu.

> **Geltungsbereich (Stand 2026-08-13):** Die Abschnitte 0–12 beschreiben die Ersteinrichtung und den
> **Hauptversuch** (2×2-Ablation, 12 Runs) in `experiments/`. Die drei Folgestudien haben eigene Ordner
> mit eigenen Skripten und werden in Abschnitt 13 behandelt. Insgesamt liegen inzwischen **65
> Produktionsruns** über vier Ordner vor (12 der 2×2-Ablation + 8 neue des $d_c$-Sweeps + 9 Layout +
> 12 MoE + 24 im 2×2×2), rund 155 GPU-Stunden. Alle Zeit- und Durchsatzangaben in diesem Dokument sind auf die tatsächlich
> gemessenen Werte aktualisiert. Das aktuelle Runbook für die Zielhardware ist
> `deepseek_mechanism_2x2x2/RTX5070_CHECKLIST.md`.

## 0. Voraussetzungen prüfen

```bash
# GPU sichtbar?
nvidia-smi

# Erwartet: NVIDIA GeForce RTX 5070, Driver >= 555.xx (besser >= 570 für Blackwell)
# CUDA-Version oben rechts sollte >= 12.8 anzeigen
```

Falls Treiber < 555 → Treiber-Update zuerst. Die RTX 5070 ist Blackwell (Compute Capability sm_120) und braucht einen modernen Treiber.

Außerdem nötig:
- **Python 3.11 oder 3.12** (3.13 vermeiden — PyTorch-Wheels noch lückenhaft)
- **~50 GB freier Plattenplatz**: ~1 GB für Daten, ~5 GB für die 12 Checkpoints des Hauptversuchs (~400 MB pro Run), Rest Puffer. Jede Folgestudie schreibt ihre Checkpoints in den eigenen Ordner und braucht dieselbe Größenordnung noch einmal (Layout 9, MoE 12, 2×2×2 24 Runs; die MoE-Modelle sind wegen der Experten größer). Checkpoints werden nicht versioniert und können nach `analyze` gelöscht werden.
- **HuggingFace-Account** mit akzeptierter Nutzungsbedingung für FineWeb-Edu (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

## 1. Repository klonen

```bash
git clone <repo-url> SeminarML
cd SeminarML/experiments
```

## 2. Python-Environment

```bash
# venv im Projektordner anlegen
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell

# pip up to date
python -m pip install --upgrade pip
```

## 3. PyTorch (Blackwell-tauglich)

**Wichtig:** RTX 5070 (sm_120) braucht PyTorch ≥ 2.6 mit CUDA 12.8. Ältere Wheels brechen ab oder fallen auf CPU zurück.

```bash
# CUDA 12.8 nightly/stable Wheel — Stand: prüfen unter https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verifizieren:

```bash
python -c "
import torch
print('Torch:',  torch.__version__)
print('CUDA :',  torch.version.cuda)
print('GPU  :',  torch.cuda.get_device_name(0))
print('Cap  :',  torch.cuda.get_device_capability(0))
print('bf16 :',  torch.cuda.is_bf16_supported())
"
```

Erwartet:
- `Torch: 2.6.x` oder neuer
- `CUDA : 12.8`
- `GPU  : NVIDIA GeForce RTX 5070`
- `Cap  : (12, 0)`  ← sm_120 Blackwell
- `bf16 : True`

Wenn `Cap` etwas anderes liefert oder `torch.cuda.is_available()` False ist → Treiber/Wheel-Kombination falsch.

## 4. Restliche Dependencies

```bash
pip install -r requirements.txt
```

## 5. HuggingFace-Login

FineWeb-Edu verlangt akzeptierte Lizenz und ist gated:

```bash
# Auf https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu einmalig „Access repository" klicken
huggingface-cli login
# Token aus https://huggingface.co/settings/tokens (Scope: „read")
```

## 6. Daten vorbereiten (einmalig, ~20–40 min)

```bash
cd experiments
python data/fineweb_edu/prepare.py
```

Erzeugt:
- `data/fineweb_edu/train.bin`  (~1 GB, 500M GPT-2-Tokens)
- `data/fineweb_edu/val.bin`    (~10 MB, 5M Tokens)

Tipp: läuft im Hintergrund in einem `tmux`/`screen`, dann sind 30 Minuten kein Problem.

## 7. Korrektheits-Tests (vor jedem Production-Run)

```bash
python test_correctness.py
```

Erwartet:

```
Running on: cuda
Test 1: Forward pass shapes ...        PASSED
Test 2: No NaN in loss for 20 random batches ... PASSED
Test 3: Parameter counts ...           PASSED
  mha         :    XX,XXX,XXX
  mha_rope    :    XX,XXX,XXX
  mla_norope  :    XX,XXX,XXX
  mla         :    XX,XXX,XXX
Test 4: Loss decreases after a few gradient steps ... PASSED
Test 5: Checkpoint save/load produces identical outputs ... PASSED
Test 6: RoPE rotation is position-dependent ... PASSED
Test 7: MHA and MLA produce different outputs ... PASSED
Test 8: Full design-doc config ...     PASSED
All tests passed.
```

Falls ein Test fehlschlägt → **nicht starten**, erst Bug fixen.

Diese Erwartungsausgabe gilt für die vier Bedingungen des Hauptversuchs (`mha`, `mha_rope`, `mla_norope`,
`mla`). Die Folgestudien haben eigene Testskripte und **andere Bedingungsnamen** — im Layout-Follow-up
`mha` / `mla_current` / `mla_deepseek`, im 2×2×2 `mha` / `mha_decoupled` / `mla_coupled` / `mla_decoupled`
je Backbone (siehe Abschnitt 13).

## 8. Smoke-Test (1 Probe-Run, ~10 min)

Vor den vollen Produktionsruns — real ~1,7–2,0 h je Dense-Run und ~3,0 h je MoE-Run — einmal mit
reduzierter Schrittzahl testen, ob alles durchläuft:

```bash
python train.py \
    --attn_mode=mha \
    --seed=42 \
    --out_dir=results/_smoke_test \
    --max_iters=300 \
    --eval_interval=50 \
    --warmup_iters=50
```

Was du sehen willst:
- `Device: cuda` (nicht cpu)
- `torch.compile: enabled` (oder eine Warnung, die einmalig akzeptiert wird; falls Compile dauerhaft fehlt → mit `--no-compile` weiterfahren)
- Loss fällt von ~10.8 (random) Richtung ~7–8 innerhalb 300 Schritten
- Keine NaN/Inf-Warnungen
- VRAM-Auslastung in `nvidia-smi` ~4–6 GB

Nach dem Smoke-Test:

```bash
rm -rf results/_smoke_test
```

## 9. Throughput-Check (optional, 1 min)

Aus dem Smoke-Test-Output die Zeile `[iter 500/...]` lesen. Pro 500 Iterationen wurden 500 × 16 × 4 × 512 ≈ 16,4M Tokens verarbeitet.

**Referenzwerte, die auf dieser Maschine tatsächlich gemessen wurden** (`deepseek_mechanism_2x2x2/plots/throughput_by_run.csv`, bf16, ohne `torch.compile`, Sequenzlänge 512):

| Backbone | gemessener Durchsatz | Zeit für 501M Tokens |
|---|---:|---:|
| Dense (6 Layer, 512) | 69.000–74.000 Tokens/s | ~1,9–2,0 h pro Run |
| MoE (2 shared + 16 routed, Top-2) | 44.000–46.000 Tokens/s | ~2,9–3,1 h pro Run |

Die Dense-Runs des Hauptversuchs und des Sweeps lagen entsprechend bei 1,71–1,88 h. Liegt der gemessene Durchsatz deutlich darunter — etwa bei 10–20K Tokens/s —, läuft etwas falsch (CPU-Fallback, fehlendes bf16, Fremdlast), und das Run-Budget muss überdacht werden, bevor eine Serie gestartet wird.

Falls `torch.compile` nicht funktioniert (Blackwell + Triton kann zickig sein), kostet das oft 30–50 % Geschwindigkeit. Workaround: Flag `--no-compile` setzen — Korrektheit bleibt unverändert.

## 10. Volles Experiment starten

```bash
# Headless im tmux/screen, damit du den Rechner nicht offen lassen musst
tmux new -s mla
cd ~/SeminarML/experiments
source .venv/bin/activate
bash run_experiments.sh 2>&1 | tee run.log
# Detach: Ctrl-b d
# Attach später: tmux attach -t mla
```

Die 12 Runs laufen sequentiell. Gemessene Wall-Clock-Zeit: **~21 h** (1,71–1,77 h pro Run, vgl. `RESULTS.md` §3). Bei Unterbrechung (Stromausfall, Reboot) einfach `bash run_experiments.sh` nochmal — abgeschlossene Runs werden über die `DONE`-Flag-Dateien übersprungen, laufende Runs setzen vom letzten Checkpoint fort.

## 11. Ergebnisse auswerten

Nach Abschluss aller 12 Runs (oder auch zwischendurch):

```bash
python analyze_results.py --results_dir results --out_dir plots
```

Output:
- `plots/learning_curves.png`
- `plots/final_val_loss.png`
- `plots/heatmap_2x2.png`
- `plots/final_val_loss_summary.csv`
- ANOVA-Tabelle und Effektgrößen auf stdout

## 12. Daten zurück zum Mac übertragen

Für Plots und Paper-Schreiben reicht die `results/`-Verzeichnisstruktur ohne die Checkpoints:

```bash
# Auf dem Desktop:
tar --exclude='checkpoint.pt' -czf results_no_ckpt.tar.gz results plots

# Auf dem Mac:
scp desktop:~/SeminarML/experiments/results_no_ckpt.tar.gz .
tar -xzf results_no_ckpt.tar.gz
```

CSV-Metriken (`metrics.csv`) sind klein — die kompletten Loss-Kurven passen in <1 MB. Die Checkpoints (je ~400 MB) lässt du auf dem Desktop, falls du noch Inferenz-Demos darauf laufen lässt.

## 13. Die drei Folgestudien

Jede Folgestudie liegt in einem eigenen Ordner unter `experiments/`, hat eigenen Modell- und Trainingscode,
eigene `results/`, eigene `plots/` und ein eigenes `RESULTS.md`. Sie teilen sich lediglich die
virtuelle Umgebung `experiments\.venv` und die tokenisierten Daten unter `experiments\data\fineweb_edu\`.
**Alle drei sind abgeschlossen** — die Befehle hier dienen der Reproduktion, nicht einem noch ausstehenden Lauf.

| Ordner | Umfang | Status | Ergebnisse |
|---|---|---|---|
| `paper_layout_followup/` | 3 Bedingungen × 3 Seeds = 9 Runs, ~16,3 GPU-h | abgeschlossen 2026-08-07 | `paper_layout_followup/RESULTS.md` |
| `moe_followup/` | 4 Zellen × 3 Seeds = 12 Runs, ~45,0 GPU-h | abgeschlossen 2026-08-09 | `moe_followup/RESULTS.md` |
| `deepseek_mechanism_2x2x2/` | 2 Backbones × 4 Zellen × 3 Seeds = 24 Runs, 59,0 GPU-h | abgeschlossen 2026-08-13 | `deepseek_mechanism_2x2x2/RESULTS.md` |

Ablauf je Ordner (PowerShell, aus dem jeweiligen Unterordner heraus):

```powershell
cd C:\Pfad\zu\SeminarML\experiments\paper_layout_followup   # bzw. moe_followup
.\run_tests.ps1          # CPU-/CUDA-unabhängige Struktur- und Referenztests
.\run_smoke_test.ps1     # kurzer GPU-Vorlauf, schreibt nur nach smoke_results/
.\run_experiments.ps1    # Produktionsruns in balancierter Reihenfolge
.\analyze.ps1            # Statistik, CSVs, Plots, RESULTS_GENERATED.md
```

Das 2×2×2 hat einen zusätzlichen Schritt: `benchmark_gpu.py` misst Micro-Batch, Gradient Accumulation,
MoE-Dispatch und `compile` auf der Zielhardware aus und schreibt `gpu_profile.json`, das alle 24 Runs
dann unverändert verwenden.

```powershell
cd C:\Pfad\zu\SeminarML\experiments\deepseek_mechanism_2x2x2
.\run_tests.ps1
.\run_smoke_test.ps1
..\.venv\Scripts\python.exe benchmark_gpu.py
.\run_experiments.ps1
.\analyze.ps1
```

Für dieses Experiment ist `deepseek_mechanism_2x2x2/RTX5070_CHECKLIST.md` das maßgebliche Runbook —
es ist neuer als dieses Dokument und deckt die Hardware-spezifischen Schritte vollständig ab.

Alle `run_experiments.ps1` sind idempotent: Runs mit `DONE`-Flag werden übersprungen, unterbrochene Runs
setzen am letzten Checkpoint fort. Einzelne Runs lassen sich gezielt starten, z. B.
`.\run_experiments.ps1 -Mode mla_deepseek -Seed 42`.

---

## Troubleshooting

**`CUDA error: no kernel image available for execution on the device`**
→ PyTorch-Wheel ohne sm_120-Support. Wheel neu installieren mit `--index-url https://download.pytorch.org/whl/cu128`. Bei nightlys ggf. auch `https://download.pytorch.org/whl/nightly/cu128`.

**`torch.compile` hängt oder bricht ab**
→ Mit `--no-compile` weiterfahren. 30–50 % langsamer, aber stabil. Compile-Bugs auf brandneuen GPUs sind in PyTorch 2.6/2.7 noch nicht selten.

**`OutOfMemoryError`**
→ Sollte bei batch=16, seq=512 nicht passieren (max ~6 GB von 12 GB). Falls doch: `--batch_size=8 --grad_accum=8` setzen (effektive Batch bleibt 64).

**`huggingface_hub.utils._errors.GatedRepoError`**
→ FineWeb-Edu-Lizenz auf der Webseite akzeptieren, dann `huggingface-cli login` mit gültigem Token.

**Loss explodiert zu NaN**
→ Sehr selten in bf16. Falls doch: `--dtype=float32` (langsamer, aber robust), oder `--grad_clip=0.5`. Vorher in `test_correctness.py` Test 2 prüfen, ob das nur bei einer Bedingung passiert.

**Training scheint zu langsam**
→ Erst prüfen: läuft auf GPU? `nvidia-smi` Auslastung >80 %? bf16 aktiv (`--dtype=bfloat16`)? Referenz sind die in Abschnitt 9 genannten ~70K Tokens/s (Dense) bzw. ~45K Tokens/s (MoE) ohne `torch.compile`. Liegt der Durchsatz weit darunter (Größenordnung 10–20K), stimmt etwas mit Device, Präzision oder Fremdlast nicht. Alle versionierten Produktionsruns liefen bewusst mit `--no-compile`, damit die Bedingungen vergleichbar bleiben.

---

## Was du auf dem Mac NICHT brauchst

Auf dem Mac (Arbeitsumgebung für Schreiben/Analyse) musst du PyTorch und CUDA nicht installieren. Es reicht die `.venv` mit:

```bash
pip install numpy pandas matplotlib scipy statsmodels
```

Damit kannst du `analyze_results.py` auch lokal laufen lassen, sobald die CSVs vom Desktop kopiert sind.
