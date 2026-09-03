# Follow-up: Sensitivität gegenüber einem DeepSeek-näheren MLA-Layout

**Status:** abgeschlossen. Alle neun Produktionsruns (3 Bedingungen × 3 Seeds, ~16,3 GPU-Stunden) sind gelaufen und am 2026-08-07 ausgewertet; die Ergebnisse stehen in [`RESULTS.md`](https://github.com/TimAlexanderStrauss/SeminarML/blob/main/experiments/paper_layout_followup/RESULTS.md) (Hauptrepo) in diesem Ordner. Dieser Ordner ist bewusst vom ursprünglichen Experiment in `experiments/` getrennt. Alte Skripte, Rohdaten und Ergebnisse werden weder überschrieben noch in die neue Auswertung gemischt.

Kurzbefund: H1 wird **nicht** gestützt — das DeepSeek-nähere Layout liegt gepaart +0,0069 Loss über der bisherigen MLA-Variante (3/3 Seeds, gerichteter Test p = 0,926); H2 ist bestätigt, `mla_deepseek` liegt +0,0253 über MHA (p = 0,026). Der übrige Text dieser Datei ist die **vor** den Trainingsläufen festgelegte Vorregistrierung und bleibt inhaltlich unverändert; die Schnellstart- und Runbook-Abschnitte beschreiben, wie die Studie reproduziert wird.

## Schnellstart auf dem Windows-PC

Nachdem das aktualisierte Repository auf dem Windows-PC angekommen ist und `experiments\.venv` sowie die FineWeb-Edu-`.bin`-Dateien vorhanden sind:

```powershell
cd C:\Pfad\zu\SeminarML\experiments\paper_layout_followup
.\run_tests.ps1
.\run_smoke_test.ps1
.\run_experiments.ps1
```

Nach Abschluss aller neun Produktionsruns:

```powershell
.\analyze.ps1
```

Die ausführliche Einrichtung und Fehlerbehebung steht in den Abschnitten 8–11.

## 1. Warum dieses Follow-up durchgeführt wird

Die ursprüngliche 2×2-Ablation operationalisiert volles MLA mit einer festen Q/K-Head-Dimension von 64:

\[
32\text{ Content-Dimensionen}+32\text{ RoPE-Dimensionen}=64.
\]

Außerdem erzeugt sie einen eigenen RoPE-Key $K^R$ für jeden der acht Heads. Diese Entscheidungen waren sinnvoll, um das 2×2 dimensionssymmetrisch zur MHA-/Partial-RoPE-Kontrollbedingung zu halten. Sie entsprechen aber nicht exakt dem Layout aus DeepSeek-V2.

DeepSeek behält die vollständige Content-Dimension und fügt den RoPE-Anteil hinzu:

\[
64\text{ Content-Dimensionen}+32\text{ RoPE-Dimensionen}=96.
\]

Der RoPE-Key $K^R$ wird außerdem zwischen allen Heads geteilt. Das Follow-up prüft, ob diese beiden Layoutentscheidungen erklären können, warum das bisherige volle MLA hinter MHA blieb.

Das ursprüngliche Experiment bleibt die primäre **Komponentenablation**. Dieses Follow-up ist eine separat geplante **Layout-Sensitivitätsanalyse**.

## 2. Forschungsfragen und vorab festgelegte Hypothesen

### Primäre Forschungsfrage

Erreicht ein DeepSeek-näheres MLA-Layout eine niedrigere finale Validation Loss als die bisherige dimensionsgeteilte MLA-Variante?

### Sekundäre Forschungsfrage

Erreicht das DeepSeek-nähere MLA die MHA-Baseline oder übertrifft es sie im untersuchten 50M-Regime?

### Hypothesen

- **H1, primär und gerichtet:** `mla_deepseek` hat eine niedrigere finale Validation Loss als `mla_current`.
- **H2, sekundär und zweiseitig:** `mla_deepseek` und `mha` unterscheiden sich in der finalen Validation Loss.

H1 ist gerichtet, weil die vollständige Content-Dimension die im ursprünglichen MLA möglicherweise verlorene Repräsentationskapazität wiederherstellt. H2 bleibt zweiseitig: Aus der kleinen Skala folgt keine belastbare Vorhersage, ob paper-näheres MLA MHA übertrifft oder weiterhin dahinter liegt.

Die Hypothesen stehen vor den Trainingsläufen in dieser Datei und sollen nach Sichtung der Ergebnisse nicht umformuliert werden.

## 3. Die drei Bedingungen

| Modus | Q/K-Aufbau je Head | RoPE-Key | Value-Dimension | Q-Kompression |
|---|---|---|---:|---|
| `mha` | 64 Dimensionen, volles RoPE | je Head | 64 | nein |
| `mla_current` | 32 Content + 32 RoPE = 64 | je Head | 64 | $d_c^Q=192$ |
| `mla_deepseek` | 64 Content + 32 RoPE = 96 | **ein geteilter Key** | 64 | $d_c^Q=192$ |

Bei der vollständigen Konfiguration ergeben sich:

| Modus | Parameter | theoretischer KV-Cache pro Token/Layer | Reduktion ggü. MHA |
|---|---:|---:|---:|
| `mha` | 44.612.608 | 1024 | 1× |
| `mla_current` | 42.845.056 | 384 | 2,67× |
| `mla_deepseek` | 42.648.448 | 160 | 6,4× |

Das DeepSeek-nähere MLA hat trotz der größeren Q/K-Dimension etwas weniger Parameter als `mla_current`: Der einzige geteilte RoPE-Key spart mehr Projektionsparameter ein, als der größere Content- und Q-RoPE-Pfad hinzufügt.

### `mha`

Unveränderte MHA-Baseline aus dem Hauptversuch. Q, K und V besitzen je Head 64 Dimensionen. RoPE wird auf alle Q/K-Dimensionen angewandt.

### `mla_current`

Entspricht der bisherigen `mla`-Bedingung:

- $c^{KV}\in\mathbb R^{128}$,
- $c^Q\in\mathbb R^{192}$,
- 32 Content- und 32 RoPE-Dimensionen,
- ein eigener $K_i^R\in\mathbb R^{32}$ je Head,
- theoretischer Cache: $128+8\cdot32=384$ Werte pro Token und Layer.

### `mla_deepseek`

Paper-nähere Variante:

- $c^{KV}\in\mathbb R^{128}$,
- $c^Q\in\mathbb R^{192}$,
- 64 vollständige Content-Dimensionen,
- 32 zusätzlich angehängte RoPE-Dimensionen,
- ein gemeinsamer $K^R\in\mathbb R^{32}$, der für alle Heads verwendet wird,
- Q/K-Dimension 96, Value-Dimension weiterhin 64,
- Attention-Skalierung automatisch mit $1/\sqrt{96}$,
- theoretischer Cache: $128+32=160$ Werte pro Token und Layer.

Die Q-Kompression bleibt absichtlich erhalten. Dadurch verändert der direkte Vergleich `mla_deepseek` gegen `mla_current` nur das Content-/RoPE-Layout und das Sharing von $K^R$, nicht zusätzlich den Q-Flaschenhals.

## 4. Was „DeepSeek-näher“ bedeutet – und was nicht

Die neue Bedingung übernimmt zwei zentrale MLA-Eigenschaften aus DeepSeek-V2:

1. RoPE-Dimensionen werden zusätzlich an die Content-Dimensionen angehängt.
2. Der RoPE-Key wird zwischen Heads geteilt.

Sie ist trotzdem keine Reproduktion des vollständigen DeepSeek-V2:

- 6-Layer-Dense-GPT statt großem MoE-Modell,
- etwa 43–45 Mio. statt Milliarden Parameter,
- 512 statt langer Kontexte,
- 501 Mio. statt hunderten Milliarden oder Billionen Trainingstokens,
- keine implementierte Weight Absorption und kein realer Inferenz-KV-Cache,
- $d_c=128$ wird zur Isolation des Layout-Effekts beibehalten.

DeepSeek-V2 komprimiert Queries; DeepSeek-V2-Lite tut dies laut Paper nicht. Dieses Follow-up behält die Q-Kompression bei und orientiert sich damit beim Q-Pfad am großen DeepSeek-V2. Eine zusätzliche No-Q-Compression-Bedingung wäre ein eigenes Experiment.

## 5. Fairness und kontrollierte Zufälligkeit

Das ursprüngliche Training nutzte denselben globalen PyTorch-Zufallszahlengenerator für Modellinitialisierung und Batch-Auswahl. Wegen unterschiedlicher Parameterformen führte derselbe Seed dadurch nicht bei allen Architekturen zu denselben Batchpositionen.

Dieses Follow-up trennt drei Zufallsquellen:

- Modellinitialisierung: `seed`,
- Trainingsdaten: `train_data_seed = 100000 + seed`,
- Validationsdaten: `val_data_seed = 200000 + seed`.

Training und Evaluation besitzen eigene `torch.Generator`-Objekte. Für Seed 42 sehen daher alle drei Bedingungen exakt dieselben zufällig gezogenen Trainings- und Validationsausschnitte. Die Generatorzustände werden im Checkpoint gespeichert, sodass auch ein fortgesetzter Run dieselbe Sequenz beibehält.

Die drei Bedingungen werden in balancierter Reihenfolge gestartet: Jede Bedingung steht einmal früh, einmal mittig und einmal spät im neunteiligen Laufplan. So ist auch ein möglicher zeitlicher Hardware-/Temperaturtrend nicht vollständig mit einer Architektur konfundiert.

## 6. Gemeinsames Trainingssetup

| Einstellung | Wert |
|---|---:|
| Layer | 6 |
| Hidden Dimension | 512 |
| Heads | 8 |
| Value-/MHA-Head-Dimension | 64 |
| Kontextlänge | 512 |
| Vokabular | GPT-2-BPE, 50.257 |
| KV-Latent $d_c$ | 128 |
| Q-Latent $d_c^Q$ | 192 |
| RoPE-Dimension $d_h^R$ | 32 |
| Iterationen | 15.300 |
| Micro-Batch | 16 |
| Gradient Accumulation | 4 |
| Effektive Batchgröße | 64 |
| Trainingstokens je Run | 501.350.400 |
| Optimizer | AdamW, Betas (0,9; 0,95) |
| Weight Decay | 0,1 |
| Learning Rate | $6\cdot10^{-4}\rightarrow6\cdot10^{-5}$ |
| Warmup | 2.000 Iterationen |
| Schedule | Cosine Decay |
| Precision | bf16 |
| Gradient Clipping | 1,0 |
| Evaluation | alle 500 Iterationen, 600 Micro-Batches |
| Finaler Endwert | Mittel der letzten 5 Evaluationen |
| Seeds | 42, 123, 456 |
| `torch.compile` | deaktiviert |

Dataset und binäre Dateien werden aus `../data/fineweb_edu/` wiederverwendet. Die neue Studie schreibt ausschließlich in ihren lokalen Ordner `paper_layout_followup/results/`.

## 7. Bewertungs- und Analysetechniken

### Validation Cross-Entropy und Perplexität

Primärmetrik ist die autoregressive Next-Token-Cross-Entropy auf dem festen FineWeb-Edu-Validationssplit. Niedriger ist besser. Perplexität wird als

\[
\operatorname{PPL}=e^{\mathcal L}
\]

berechnet.

Die finale Validation Loss eines Runs ist das Mittel der letzten fünf Evaluationen. Dadurch wird das Samplingrauschen eines einzelnen Endpunkts reduziert. Die statistische Replikationsebene bleiben die drei Seeds.

### Gepaarte Vergleiche

Weil alle Bedingungen je Seed dieselben Datenfenster sehen, werden die Vergleiche seedweise gepaart.

Primärer Kontrast:

\[
\Delta_{primary}=L_{mla\_deepseek}-L_{mla\_current}.
\]

Ein negatives Delta stützt H1. Dafür wird zusätzlich zur mittleren gepaarten Differenz ein gerichteter gepaarter t-Test berichtet.

Sekundärer Kontrast:

\[
\Delta_{secondary}=L_{mla\_deepseek}-L_{mha}.
\]

Dieser Vergleich ist zweiseitig. Wegen nur drei Seed-Paaren sind Effektgröße, Konfidenzintervall und die Konsistenz der Einzelseeds wichtiger als ein isolierter p-Wert.

### Abbildungen

`analyze_results.py` erzeugt:

- `learning_curves.png`: Training und Validation, Mittel ± Seed-Standardabweichung,
- `final_val_loss.png`: Endwerte, 95-%-CI und verbundene Seed-Punkte,
- `paired_differences.png`: gepaarte Differenzen je Seed,
- `quality_vs_cache.png`: Modellqualität gegen theoretische Cache-Größe,
- CSV-Tabellen und einen automatisch erzeugten Ergebnisbericht.

## 8. Windows-Vorbereitung

Die folgenden Befehle werden in **PowerShell** ausgeführt.

### 8.1 In das bestehende Experiment-Verzeichnis wechseln

```powershell
cd C:\Pfad\zu\SeminarML\experiments
```

### 8.2 Virtuelle Umgebung prüfen oder anlegen

Falls `experiments\.venv` vom ursprünglichen Experiment noch existiert, kann sie weiterverwendet werden.

Falls nicht:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Falls PowerShell die Aktivierung blockiert:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

PyTorch für die RTX 5070 entsprechend der installierten CUDA-/Treiberumgebung installieren. Die bereits für das Hauptprojekt funktionierende PyTorch-Installation sollte bevorzugt weiterverwendet werden. Danach:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Die Ausgabe muss `True` und deine RTX 5070 zeigen.

### 8.3 Daten prüfen

Diese Dateien müssen vorhanden sein:

```text
experiments\data\fineweb_edu\train.bin
experiments\data\fineweb_edu\val.bin
```

Falls sie auf dem Windows-PC noch fehlen:

```powershell
.\.venv\Scripts\python.exe data\fineweb_edu\prepare.py
```

Der Download beziehungsweise die Tokenisierung wird nur einmal benötigt.

## 9. Tests vor dem Training

In den neuen Ordner wechseln:

```powershell
cd paper_layout_followup
```

CPU-/CUDA-unabhängige Struktur- und Referenztests:

```powershell
.\run_tests.ps1
```

Erwartetes Ende:

```text
All correctness tests passed.
All numerical reference tests passed.
All follow-up tests passed.
```

Danach kurzer GPU-Smoke-Test für alle drei Bedingungen:

```powershell
.\run_smoke_test.ps1
```

Dieser schreibt ausschließlich nach `smoke_results/` und gehört nicht zur späteren Analyse.

## 10. Vollständige Experimente starten

Alle neun Runs in balancierter Reihenfolge:

```powershell
.\run_experiments.ps1
```

Das Skript führt aus:

```text
mha          seed 42
mla_current  seed 42
mla_deepseek seed 42
mla_current  seed 123
mla_deepseek seed 123
mha          seed 123
mla_deepseek seed 456
mha          seed 456
mla_current  seed 456
```

Einzelne Bedingung oder einzelner Run:

```powershell
.\run_experiments.ps1 -Mode mla_deepseek
.\run_experiments.ps1 -Mode mla_deepseek -Seed 42
```

Unterbrochene Runs setzen automatisch am letzten Checkpoint fort. Abgeschlossene Runs besitzen eine `DONE`-Datei und werden beim erneuten Aufruf übersprungen.

Auf Grundlage der bisherigen Laufzeiten sollte mit ungefähr 16–20 GPU-Stunden für alle neun Runs gerechnet werden. Die 96-dimensionalen Q/K-Tensoren von `mla_deepseek` können etwas langsamer sein als das bisherige MLA.

## 11. Ergebnisse analysieren

Nach Abschluss aller neun Runs:

```powershell
.\analyze.ps1
```

Alternativ:

```powershell
..\.venv\Scripts\python.exe analyze_results.py
```

Die Ergebnisse erscheinen unter `plots/`. Besonders wichtig:

- `plots\RESULTS_GENERATED.md`,
- `plots\final_val_loss.png`,
- `plots\paired_differences.png`,
- `plots\quality_vs_cache.png`,
- `plots\paired_contrasts.csv`.

Das Analyseskript überprüft vor der Statistik automatisch:

- keine doppelten Modus-/Seed-Kombinationen,
- identische kontrollierte Hyperparameter,
- identische Trainings- und Validation-Datenseeds je Seed,
- vollständige finale Evaluationspunkte,
- vorhandene beziehungsweise fehlende `DONE`-Flags.

## 12. Interpretation der möglichen Ergebnisse

### `mla_deepseek` ist klar besser als `mla_current`

Dann hat das dimensionsgeteilte Layout das bisherige volle MLA wahrscheinlich benachteiligt. Je nach Abstand zu MHA wäre der ursprüngliche Nicht-Replikationsbefund teilweise oder weitgehend layoutabhängig.

### `mla_deepseek` erreicht ungefähr MHA

Dann wäre die schwächere DeepSeek-Aussage gestützt: starke Cache-Kompression kann bei geeigneterem Layout nahezu ohne Qualitätsverlust funktionieren. Ein echter Qualitätsgewinn wäre weiterhin nicht gezeigt.

### `mla_deepseek` schlägt MHA

Das wäre ein starker Befund, aber nicht automatisch ein reiner Decoupled-RoPE-Effekt. `mla_deepseek` besitzt 96 statt 64 Q/K-Dimensionen. Zusätzliche Q/K-Kapazität ist deshalb eine Alternativerklärung und muss gemeinsam mit dem besseren Cache-Layout diskutiert werden.

### `mla_deepseek` verbessert sich nicht

Dann ist die Layoutabweichung wahrscheinlich nicht die Hauptursache. Skala, KV-Latent-Dimension, Q-Kompression, Optimierung oder andere DeepSeek-Komponenten werden als Erklärungen wichtiger.

## 13. Dateien in diesem Ordner

| Datei | Zweck |
|---|---|
| `model.py` | drei Attention-Bedingungen |
| `train.py` | RNG-getrenntes Training und checkpointbarer Datenstrom |
| `run_experiments.ps1` | neun Windows-Produktionsruns |
| `run_tests.ps1` | strukturelle und numerische Tests |
| `run_smoke_test.ps1` | kurzer GPU-Vorlauf |
| `analyze.ps1` | Windows-Wrapper für die Auswertung |
| `analyze_results.py` | Validierung, gepaarte Statistik, CSV und Plots |
| `test_correctness.py` | Shapes, Gradienten, Layout, Sharing, RNG und Parameterzahlen |
| `test_reference.py` | Vergleich mit naiver Attention-Referenz |
| `requirements.txt` | verweist auf die Requirements des Hauptversuchs |

Die ursprünglichen Ergebnisse bleiben unter `../results/`, `../results_sweep/` und `../plots/` unverändert erhalten.
