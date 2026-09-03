# DeepSeek-nahe MLA-Mechanik: gemeinsames 2×2×2-Experiment

**Status:** Abgeschlossen. Alle 24 Produktionsruns sind auf der RTX 5070 gelaufen (59,0 GPU-Stunden bis zur letzten Evaluation bei Iteration 15.000) und ausgewertet; die Designprüfung meldet 24 vollständige Runs mit `DONE`-Flag und keine Warnung. Der vollständige Ergebnisbericht steht in [`RESULTS.md`](https://github.com/TimAlexanderStrauss/SeminarML/blob/main/experiments/deepseek_mechanism_2x2x2/RESULTS.md) (Hauptrepo), der automatisch erzeugte Kurzbericht in `plots/RESULTS_GENERATED.md` (wird von der Auswertung erzeugt).

Kurzfassung: Von den drei Faktoren übersteht auf der Attention-Achse nur die Low-Rank-KV-Kompression die Holm-Korrektur (+0,0409 Loss, Holm-*p* = 0,033). Der entkoppelte RoPE-Pfad ist nicht nachweisbar (−0,0082, Holm-*p* = 0,35). Der MoE-Backbone senkt die Loss um 0,0787 (Holm-*p* = 0,005). Keine der vier Interaktionen ist nach Holm signifikant; die Richtung aller Effekte ist zwischen Dense und MoE identisch.

Die vollständige Prüfung auf der Zielhardware steht in [`RTX5070_CHECKLIST.md`](RTX5070_CHECKLIST.md).

Dieser Ordner ist unabhängig von allen bisherigen Experimenten. Er schreibt nur nach `deepseek_mechanism_2x2x2/results/` und `deepseek_mechanism_2x2x2/plots/`.

## 1. Forschungsfrage

Das Experiment untersucht diese Frage:

> Welcher MLA-Bestandteil verändert die Modellqualität in einer DeepSeek-nahen Attention-Architektur: die gemeinsame Low-Rank-KV-Kompression, der entkoppelte RoPE-Pfad oder ihre Interaktion? Ändert sich der Befund zwischen einem Dense- und einem MoE-Backbone?

Das DeepSeek-V2-Paper vergleicht vollständiges MLA mit MHA. Dieser Vergleich verwendet MoE-Modelle und trennt die beiden MLA-Bestandteile nicht. Dieses Experiment ergänzt deshalb ein faktorielles Design. Es ist keine Reproduktion der Modellgröße oder des Trainingsbudgets aus dem Paper.

Primärquelle: [DeepSeek-V2, Abschnitt 2.1 und Appendix D.2](https://arxiv.org/html/2405.04434).

## 2. Design

Das Design hat drei Faktoren mit je zwei Stufen:

| Faktor | Stufe 0 | Stufe 1 |
|---|---|---|
| KV-Projektion | vollrangige K-/V-Projektionen | gemeinsame Low-Rank-KV-Kompression |
| RoPE-Pfad | gekoppeltes RoPE auf Q und K | zusätzlicher, entkoppelter RoPE-Pfad |
| Backbone | Dense SwiGLU | skalierter DeepSeekMoE |

Es gibt acht Zellen und drei Seeds. Das ergibt 24 Runs.

| `attn_mode` | Low-Rank KV | entkoppeltes RoPE | Bedeutung |
|---|:---:|:---:|---|
| `mha` | nein | nein | MHA-Kontrolle mit normalem RoPE |
| `mha_decoupled` | nein | ja | vollrangige Kontrolle mit DeepSeek-artigem RoPE-Pfad |
| `mla_coupled` | ja | nein | Low-Rank-KV-Kontrolle mit gekoppeltem RoPE |
| `mla_decoupled` | ja | ja | vollständige DeepSeek-nahe MLA-Zelle |

Jede Attention-Zelle läuft mit beiden Backbones. Alle acht Zellen verwenden für einen Seed dieselben Trainings- und Validierungsfenster.

## 3. DeepSeek-nahe Attention

Die MLA-Zelle übernimmt die folgenden Strukturmerkmale aus DeepSeek-V2:

- Die KV-Kompression erzeugt einen gemeinsamen Latent-Vektor `c_kv`.
- `W_UK` und `W_UV` rekonstruieren Content-Keys und Values aus `c_kv`.
- Die Content-Dimension pro Head bleibt 64.
- Der entkoppelte RoPE-Pfad fügt 32 Dimensionen hinzu. Er entfernt keine Content-Dimensionen.
- Ein 32-dimensionaler RoPE-Key wird zwischen allen Heads geteilt.
- Q und K haben in den entkoppelten Zellen 96 Dimensionen. V hat 64 Dimensionen.
- PyTorch skaliert den Attention-Score deshalb mit `sqrt(96)`.
- `d_c=256` entspricht `4*d_h`, wie in DeepSeek-V2 und DeepSeek-V2-Lite.

Die Wahl von `d_c` ist bei einem verkleinerten Modell nicht eindeutig. `d_c=128`
würde mit `d_c/d_model=0,25` das Verhältnis von DeepSeek-V2-Lite erhalten. Es
entspräche hier aber nur `2*d_h` und läge in der Kompressionsdosis, für die der
frühere Sweep einen deutlichen Qualitätsverlust zeigte. `d_c=256` erhält das in
beiden offiziellen V2-Modellen gleiche Verhältnis `d_c=4*d_h`. Es macht die
vollständige MLA-Zelle außerdem fast parameterneutral zu MHA. Dafür ist der
logische MLA-Cache größer als bei `d_c=128`. Die Wahl ist eine dokumentierte
Skalierungsentscheidung und keine exakte Reproduktion aller Paper-Verhältnisse.

### Entscheidung zur Query-Kompression

Query-Kompression ist in allen Zellen deaktiviert. Diese Entscheidung hat zwei Gründe:

1. Der Faktor „Low-Rank“ soll nur die gemeinsame KV-Kompression ändern.
2. DeepSeek-V2-Lite verwendet keine Query-LoRA-Kompression. Das macht diese Wahl für das kleine Modell plausibel.

Das Experiment beantwortet damit die Frage nach **Low-Rank KV**, nicht nach einem gemeinsamen Bündel aus KV- und Query-Kompression. Eine zusätzliche Query-Kompressionsablation wäre ein eigener Faktor und würde mehr als 24 Runs erfordern.

### Bedeutung der Kontrollzellen

`mla_coupled` ist ein mechanistischer Gegenentwurf. RoPE liegt dort auf den rekonstruierten Keys. Dadurch ist die für effiziente Inferenz wichtige Weight-Absorption nicht möglich. Die Zelle ist trotzdem notwendig. Sie isoliert den Qualitätseffekt der KV-Kompression ohne den entkoppelten RoPE-Pfad.

`mha_decoupled` ist ebenfalls eine Kontrollzelle. Sie prüft den entkoppelten RoPE-Pfad ohne KV-Kompression.

Der RoPE-Faktor enthält die zusätzliche 32-dimensionale Positionskomponente. Dies entspricht der DeepSeek-Mechanik. Ein beobachteter RoPE-Effekt ist deshalb ein Effekt des gesamten entkoppelten Positionspfads. Er ist nicht nur ein Effekt der Rotation.

### Parameter und logischer Cache

Die Werte gelten für sechs Layer. „Mit Recompute“ erlaubt bei `mla_coupled`,
Content-Keys und Values für den Prefix aus `c_kv` neu zu berechnen.

| Backbone | Attention | Gesamtparameter | aktive Parameter/Token | Cache ohne Recompute | Cache mit Recompute |
|---|---|---:|---:|---:|---:|
| Dense | `mha` | 44.416.000 | 44.416.000 | 6.144 | 6.144 |
| Dense | `mha_decoupled` | 45.300.736 | 45.300.736 | 6.336 | 6.336 |
| Dense | `mla_coupled` | 43.631.104 | 43.631.104 | 6.144 | 1.536 |
| Dense | `mla_decoupled` | 44.515.840 | 44.515.840 | 1.728 | 1.728 |
| MoE | `mha` | 80.583.680 | 44.456.960 | 6.144 | 6.144 |
| MoE | `mha_decoupled` | 81.468.416 | 45.341.696 | 6.336 | 6.336 |
| MoE | `mla_coupled` | 79.798.784 | 43.672.064 | 6.144 | 1.536 |
| MoE | `mla_decoupled` | 80.683.520 | 44.556.800 | 1.728 | 1.728 |

Der Spread der aktiven Parameter beträgt 3,9 %. Die Faktoren sind damit nicht
vollständig parameterneutral. Low-Rank entfernt Parameter. Der entkoppelte
RoPE-Pfad fügt pro Layer 147.456 Parameter hinzu. Effekte sind deshalb immer
Effekte der vollständigen Architekturänderung, nicht Effekte bei exakt gleicher
Parameterzahl.

## 4. Backbones

### Gemeinsame Eigenschaften

- 6 Transformer-Layer
- Hidden-Dimension 512
- 8 Attention-Heads
- RMSNorm
- SwiGLU-FFNs
- erster Layer immer Dense
- keine gelernten Positions-Embeddings

### Dense

Alle sechs Layer verwenden einen Dense-SwiGLU-FFN mit Zwischenbreite 1344.

### MoE

Layer 0 verwendet denselben Dense-SwiGLU-FFN wie der Dense-Backbone. Layer 1 bis 5 verwenden:

- 2 Shared Experts
- 16 Routed Experts
- Top-2-Routing
- Expertbreite 336
- Softmax-Router in float32
- keine Top-k-Renormalisierung
- sequenziellen Expert-Balance-Loss mit `alpha=0.001`
- kein Token-Dropping

Pro Token sind vier Experten aktiv: zwei Shared Experts und zwei Routed Experts. Die aktive Zwischenbreite ist `4 × 336 = 1344`. Sie ist damit gleich zur Zwischenbreite des Dense-FFNs. Die aktive FFN-Matrixarbeit ist in beiden Backbones gleich.

Die Anzahl der Experten und der Top-k-Wert sind für 12 GiB VRAM skaliert. Die Strukturmerkmale bleiben erhalten: Shared-Expert-Isolation, fein aufgeteilte Routed Experts und sparsames Routing.

## 5. Vorab festgelegte Auswertung

Die Replikationseinheit ist der Seed. Die Analyse berechnet alle Effekte zuerst innerhalb eines Seeds. Danach testet sie den seed-gepaarten Kontrast mit einem zweiseitigen Ein-Stichproben-t-Test gegen null.

Die primäre Analyse enthält sieben Effekte:

1. Low-Rank KV
2. entkoppeltes RoPE
3. MoE-Backbone
4. Low-Rank KV × entkoppeltes RoPE
5. Low-Rank KV × MoE
6. entkoppeltes RoPE × MoE
7. Low-Rank KV × entkoppeltes RoPE × MoE

Das Analyseskript berichtet rohe p-Werte und eine Holm-Korrektur über diese sieben Tests. Bei nur drei Seeds sind Konfidenzintervalle, Einzelseeds und Effektgrößen wichtiger als ein einzelner p-Wert.

Die 3-Wege-Interaktion ist der statistisch schwierigste Kontrast. Sie bildet die
Differenz der beiden vollständigen 2×2-Interaktionen und mittelt nicht über
weitere Zellen. Mit drei Seeds und sieben Holm-korrigierten Tests ist ihre Power
begrenzt. Der Bericht muss deshalb Schätzwert, 95-%-Konfidenzintervall und alle
drei Seed-Kontraste zeigen. Zusätzliche Seeds wären nur als vollständige Sätze
mit allen acht Zellen gültig.

Die mechanistische Interpretation verwendet zusätzlich diese gepaarten Kontraste pro Backbone:

- entkoppeltes RoPE ohne Low-Rank KV
- entkoppeltes RoPE mit Low-Rank KV
- Low-Rank KV mit gekoppeltem RoPE
- Low-Rank KV mit entkoppeltem RoPE
- vollständiges MLA gegen MHA

Positive Differenzen bedeuten höhere Validation Loss und damit schlechtere Qualität.

Der finale Wert eines Runs ist das Mittel der letzten fünf Validation-Evaluationen. `val_loss` enthält nur Next-Token-Cross-Entropy. Der MoE-Balance-Loss wird getrennt protokolliert.

## 6. Training

| Einstellung | Wert |
|---|---:|
| Datensatz | FineWeb-Edu `sample-10BT` |
| Tokenizer | GPT-2-BPE, 50.257 Tokens |
| Kontextlänge | 512 |
| Iterationen | 15.300 |
| effektive Batchgröße | 64 Sequenzen |
| Trainingstokens pro Run | 501.350.400 |
| Seeds | 42, 123, 456 |
| Optimizer | AdamW `(0.9, 0.95)` |
| Learning Rate | `6e-4` bis `6e-5`, Cosine |
| Warmup | 2.000 Iterationen |
| Weight Decay | 0,1 |
| Gradient Clipping | 1,0 |
| Precision | bf16 |
| Evaluation | alle 500 Iterationen, 600 Batches |

Das GPU-Profil darf Micro-Batch und Gradient Accumulation ändern. Ihr Produkt muss immer 64 sein. Alle 24 Runs verwenden dasselbe Profil.

Die Evaluation verwendet dieselbe Micro-Batch-Größe wie das Training. Bei
Micro-Batch 16 umfasst jede Evaluation 4.915.200 Tokenpositionen. Bei
Micro-Batch 32 sind es 9.830.400. Alle 24 Runs bleiben intern vergleichbar. Das
Rauschniveau ist dann aber nicht direkt mit älteren Experimenten vergleichbar.

## 7. RTX-5070-Optimierung

Die alte MoE-Implementierung startete für jeden Expert drei kleine Matrixoperationen in einer Python-Schleife. Dies erzeugte viele kleine GPU-Kernels.

Der neue Standardpfad arbeitet anders:

1. Er sortiert die Zuweisungen nach Expert und berechnet kompakte Slot-Indizes.
2. Er packt ausgewählte Tokens in einen Tensor mit einer Expertendimension.
3. Er berechnet alle Expert-Gate-Projektionen mit einem batched Matrixprodukt.
4. Er berechnet alle Up-Projektionen mit einem zweiten batched Matrixprodukt.
5. Er berechnet alle Down-Projektionen mit einem dritten batched Matrixprodukt.
6. Er schreibt die gewichteten Ergebnisse zu den ursprünglichen Tokens zurück.

Dieser Pfad erzeugt keine verworfenen Tokens. Im Eager-Modus verwendet er die
tatsächlich benötigte, auf acht Slots gerundete Kapazität. Im kompilierten Modus
beendet ein datenabhängiger Assert den Run, wenn die vorab reservierte Kapazität
nicht reicht. Es verwirft keine Zuweisung. Erhöhe in diesem Fall den
Kapazitätsfaktor und starte den betroffenen Run neu. `test_correctness.py` prüft
Ausgaben und Gradienten gegen die alte Expertenschleife. Es provoziert außerdem
den Assert im kompilierten CPU-Pfad. Die Checkliste verlangt denselben Test mit
CUDA und `reduce-overhead` auf der RTX 5070.

Der batched Pfad füllt freie Expert-Slots mit Nullen. Diese Slots ändern das Ergebnis und die Gradienten nicht. Sie erhöhen aber die tatsächlich ausgeführte Kernelarbeit. Die Angabe „gleicher aktiver Compute“ beschreibt deshalb die mathematisch aktiven Experten, nicht die gepaddeten GPU-FLOPs. Verwende die Laufzeit nicht als Beleg für die allgemeine Effizienz von MoE.

`benchmark_gpu.py` testet auf der tatsächlichen GPU:

- Micro-Batch 32, 16 und bei Bedarf 8
- batched und alte Loop-Dispatch-Varianten
- Ausführung mit und ohne `torch.compile`
- effektiven Durchsatz und maximal reservierten VRAM
- den echten `np.memmap`-/`get_batch`-/Host-to-GPU-Datenpfad

Das Skript schreibt die schnellste erfolgreiche Konfiguration nach `gpu_profile.json`. Der Produktions-Runner verwendet dieses Profil für alle 24 Runs. Dies ist besser als eine feste Annahme über `torch.compile`, weil dessen Nutzen von der installierten PyTorch-, CUDA- und Triton-Version abhängt.

Der Benchmark akzeptiert standardmäßig nur Profile, die höchstens 90 % des verfügbaren VRAM reservieren. Diese Reserve schützt Evaluation, Checkpoints und den Windows-Grafiktreiber vor einem knappen VRAM-Profil.

Eine hohe Prozentzahl in `nvidia-smi` ist nicht allein das Ziel. Der entscheidende Wert ist der gemessene Durchsatz in Tokens pro Sekunde. Während des Trainings protokolliert `train.py` zusätzlich Peak-VRAM und Tokens pro Sekunde.

## 8. Ausführung auf Windows

### Tests

```powershell
cd C:\Pfad\zu\SeminarML\experiments\deepseek_mechanism_2x2x2
.\run_tests.ps1
```

### GPU-Profil erzeugen

Prüfe zuerst alle acht Zellen einmal mit CUDA und bf16:

```powershell
.\run_smoke_test.ps1
```

Erzeuge danach das Leistungsprofil:

```powershell
..\.venv\Scripts\python.exe benchmark_gpu.py
```

Der kurze Benchmark testet sechs sinnvolle Profile. Der vollständige Benchmark testet zwölf Profile:

```powershell
..\.venv\Scripts\python.exe benchmark_gpu.py --full
```

Prüfe danach `gpu_profile.json`. Der Dateiname der GPU muss deine RTX 5070 enthalten. Wenn `compile=true` ist, hat der Benchmark die kompilierte Variante tatsächlich ausgeführt und gemessen.

### Alle 24 Runs starten

```powershell
.\run_experiments.ps1
```

Der Runner überspringt abgeschlossene Runs mit `DONE`-Flag. Ein unterbrochener Run lädt automatisch seinen letzten Checkpoint.

Einzelne Teile lassen sich gezielt starten:

```powershell
.\run_experiments.ps1 -Backbone moe
.\run_experiments.ps1 -Backbone moe -Attention mla_decoupled -Seed 42
```

### Ergebnisse auswerten

```powershell
.\analyze.ps1
```

Die Analyse verweigert eine gemeinsame Auswertung, wenn kontrollierte Hyperparameter zwischen Runs abweichen. Sie meldet auch fehlende Runs und fehlende `DONE`-Flags.

## 9. Erwarteter Aufwand

Die alte Implementierung benötigte ungefähr 21 GPU-Stunden für zwölf Dense-Runs und 45 GPU-Stunden für zwölf MoE-Runs. Der neue batched MoE-Pfad soll die MoE-Zeit senken. Die genaue Zeit ist erst nach dem RTX-5070-Benchmark bekannt.

Verwende für die Planung zunächst diese konservative Spanne:

- Dense: 21 bis 25 GPU-Stunden
- MoE: 30 bis 45 GPU-Stunden
- Gesamt: 51 bis 70 GPU-Stunden

Der Benchmark liefert eine bessere Schätzung. Teile `501.350.400` Tokens durch den gemessenen Trainingsdurchsatz eines Runs. Evaluation, Checkpoints und Compiler-Start erhöhen die reale Laufzeit zusätzlich.

## 10. Aussagegrenze

Dieses Experiment kann zeigen, welcher MLA-Bestandteil die Validation Loss im untersuchten 44–81-Millionen-Parameter-Regime verändert. Es kann auch zeigen, ob der Effekt vom Dense-/MoE-Backbone abhängt.

Es kann nicht beweisen, dass derselbe Bestandteil den Benchmark-Vorteil von DeepSeek-V2 bei 16 bis 250 Milliarden Parametern verursacht. Dafür unterscheiden sich Modellgröße, Trainingsdaten, Tokenbudget und Evaluation zu stark. Formuliere das Ergebnis deshalb als lokale mechanistische Evidenz in einer skalierten, DeepSeek-nahen Architektur.

Weitere Grenzen:

- Die Architektur ist vollständig in den untersuchten Achsen Attention-Layout
  und Backbone-Typ. Sie ist keine vollständige DeepSeek-V2-Reproduktion.
- Query-Kompression ist entsprechend DeepSeek-V2-Lite deaktiviert.
- Das MoE verwendet Top-2/16 statt Top-6/64 bei V2-Lite. Device-Limited
  Routing, Device-/Communication-Balance und Token Dropping entfallen.
- Die zusätzlichen RoPE-Dimensionen ändern Q/K-Kapazität und Score-Skalierung.
- Es gibt kein YaRN-Long-Context-Training und keinen realen autoregressiven
  KV-Cache- oder Weight-Absorption-Benchmark.
- Die Hauptanalyse hat nur drei vollständige Seeds. Besonders die
  3-Wege-Interaktion ist schwach gepowert.

## 11. Dateien

| Datei | Zweck |
|---|---|
| `model.py` | acht Modellzellen und optimierter MoE-Pfad |
| `train.py` | Training, Resume, RNG-Trennung und Logging |
| `benchmark_gpu.py` | RTX-Profil für Batch, Dispatch und Compiler |
| `smoke_test_gpu.py` | bf16-CUDA-Prüfung aller acht Zellen |
| `test_gpu_capacity.py` | erzwungener Kapazitätsfehler unter CUDA und `torch.compile` |
| `run_experiments.py` | balancierter Plan für 24 Runs |
| `run_experiments.ps1` | Windows-Einstiegspunkt |
| `analyze_results.py` | Designprüfung und gepaarte 2×2×2-Analyse |
| `test_correctness.py` | Struktur-, Gradienten-, Router- und Designprüfungen |
| `test_reference.py` | unabhängige numerische Attention-Referenz |
