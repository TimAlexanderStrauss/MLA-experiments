# Follow-up: ursprüngliche 2×2-Ablation in einem DeepSeekMoE-Backbone

**Status:** abgeschlossen. Tests, GPU-Smoke-Test und alle zwölf Produktionsruns (4 Attention-Zellen × 3 Seeds, ~45,0 GPU-Stunden) sind gelaufen und am 2026-08-09 ausgewertet; die Ergebnisse stehen in [`RESULTS.md`](https://github.com/TimAlexanderStrauss/SeminarML/blob/main/experiments/moe_followup/RESULTS.md) (Hauptrepo) in diesem Ordner. Dieser Ordner besitzt eigene Checkpoints, Rohdaten und Plots. Das seit 2026-08-07 abgeschlossene `paper_layout_followup` und die historischen Ergebnisse werden nicht verändert.

Kurzbefund: Die 2×2-Struktur bleibt im MoE-Backbone vollständig erhalten — Low-Rank schadet (+0,0383; $t(2)=11{,}85$, $p=0{,}0071$), Decoupled RoPE hilft (−0,0144; $p=0{,}0270$), die Interaktion bleibt negativ (−0,0262; $p=0{,}0033$). Der übrige Text dieser Datei ist die **vor** den Trainingsläufen festgelegte Vorregistrierung samt Runbook und bleibt inhaltlich unverändert.

## Schnellstart auf Windows

```powershell
cd C:\Pfad\zu\SeminarML\experiments\moe_followup
.\run_tests.ps1
.\run_smoke_test.ps1
.\run_experiments.ps1
```

Nach allen zwölf Produktionsruns:

```powershell
.\analyze.ps1
```

Ein einzelner Run lässt sich gezielt starten oder fortsetzen:

```powershell
.\run_experiments.ps1 -Mode mla -Seed 42
```

Vorhandene Checkpoints werden automatisch fortgesetzt; ein vorhandenes `DONE` überspringt den Run.

## 1. Forschungslogik

Der Performance-Vergleich MHA gegen MLA in DeepSeek-V2 wurde nicht in einem kleinen Dense-GPT, sondern in MoE-Modellen durchgeführt. Das erste Seminar-Experiment fand im Dense-Regime einen deutlichen Nachteil der Low-Rank-Kompression bei `d_c=128`. Dieses Follow-up prüft, ob derselbe mechanistische Befund in einem sparsamen Expert-Backbone bestehen bleibt.

Die vier Attention-Zellen bleiben exakt gleich:

| Modus | Low-Rank | Decoupled RoPE |
|---|:---:|:---:|
| `mha` | nein | nein |
| `mha_rope` | nein | ja |
| `mla_norope` | ja | nein |
| `mla` | ja | ja |

Nur der FFN-Backbone ändert sich. Dadurch ist das Experiment eine **Replikation der ursprünglichen 2×2-Komponentenablation unter MoE**, nicht eine neue Vermischung mit der getrennt durchgeführten MLA-Layout-Sensitivitätsstudie.

### Vorab festgelegte Fragen

1. Bleibt der Haupteffekt von Low-Rank-Kompression unter MoE negativ?
2. Bleibt der Haupteffekt von Decoupled RoPE positiv?
3. Ändert MoE die Interaktion zwischen beiden Komponenten?
4. Deskriptiv: Verschiebt sich die gesamte Validation Loss gegenüber dem historischen Dense-Backbone?

Die primäre Auswertung verwendet **seed-gepaarte 2×2-Kontraste innerhalb des MoE-Experiments**. Die vier Zellen teilen je Seed dieselben Datenfenster. Der Dense-vs.-MoE-Plot ist nur deskriptiv, weil das historische Experiment noch keine architekturunabhängigen Data-RNGs verwendete.

> **Nachtrag 2026-08-13:** Diese Einschränkung gilt weiterhin für den studienübergreifenden Vergleich in `RESULTS.md` (deskriptiv 0,104–0,114 Loss Gewinn je Zelle). Der Backbone-Effekt selbst ist inzwischen sauber geschätzt: Im gemeinsamen 2×2×2-Experiment ([`deepseek_mechanism_2x2x2/RESULTS.md`](https://github.com/TimAlexanderStrauss/SeminarML/blob/main/experiments/deepseek_mechanism_2x2x2/RESULTS.md) im Hauptrepo) laufen Dense und MoE seed-gepaart in **einer** Studie und liefern −0,0787 (Holm $p = 0{,}0052$). Für eine kausale Aussage über den Backbone ist dieser Wert heranzuziehen, nicht die Gegenüberstellung hier — die beiden Zahlen sind auch nicht direkt vergleichbar, weil sich Attention-Layout, Q-Kompression und Kompressionsdosis zwischen den Studien unterscheiden.

## 2. MoE-Architektur

DeepSeek-V2 ersetzt alle FFNs außer dem ersten durch DeepSeekMoE. Entsprechend gilt hier:

- Layer 0: ursprünglicher Dense-GELU-FFN;
- Layer 1–5: MoE;
- 2 immer aktive Shared Experts;
- 16 Routed Experts;
- Top-2 Routed Experts pro Token;
- SwiGLU je Expert;
- Expert-Zwischendimension 336;
- Router: lineare Centroids ohne Bias, Softmax in float32, greedy Top-k;
- ausgewählte Routergewichte werden **nicht** erneut auf Summe 1 normiert;
- sequenzweiser Expert-Balance-Loss mit `alpha=0.001`;
- keine Capacity-Limits und kein Token Dropping.

Pro Token sind 2 Shared + 2 Routed Experts aktiv. Mit drei SwiGLU-Matrizen und Breite 336 entspricht der aktive FFN-Matrixaufwand 98,44 % des bisherigen Dense-GELU-FFNs. Damit bleibt der aktive Compute annähernd konstant, während die gesamte Expertenkapazität wächst.

Vollkonfiguration:

| Modus | Gesamtparameter | aktive Parameter pro Token |
|---|---:|---:|
| `mha` | 80.616.448 | 44.489.728 |
| `mha_rope` | 80.616.448 | 44.489.728 |
| `mla_norope` | 78.259.072 | 42.132.352 |
| `mla` | 78.848.896 | 42.722.176 |

„Aktiv“ zählt die Embeddings, Attention, Router, Shared Experts und zwei ausgewählte Routed Experts. Die tatsächlich ausgewählten Routed Experts können pro Token wechseln.

## 3. Paper-Treue und dokumentierte Abweichungen

Übernommen aus DeepSeek-V2 beziehungsweise V2-Lite:

- Shared-Expert-Isolation und feinere Routed Experts;
- erste Schicht dense, alle folgenden Schichten MoE;
- SwiGLU-Experten;
- greedy Softmax-Top-k ohne Renormalisierung (`norm_topk_prob=false`);
- Expert-Level-Balance-Loss `alpha=0.001`, wie bei V2-Lite auf einem Device;
- keine Device-/Communication-Balance-Losses auf einem einzelnen Gerät.

Notwendige oder experimentell motivierte Abweichungen:

| DeepSeek-V2-Lite | Dieses Follow-up | Grund |
|---|---|---|
| 27 Layer, `d_model=2048` | 6 Layer, `d_model=512` | bestehende 50M-Seminar-Skala und RTX 5070 |
| 64 routed, Top-6, 2 shared | 16 routed, Top-2, 2 shared | weniger Speicher/Kernellaunches; sparse Konkurrenz bleibt erhalten |
| Expertbreite 1408 | Expertbreite 336 | aktiven Compute an bisherigen Dense-FFN angleichen |
| Dense Layer als DeepSeek-SwiGLU | Layer 0 bleibt bisheriger GELU-FFN | Attention-2×2 und möglichst viel des alten Backbones kontrolliert halten |
| Multi-GPU-Mechanismen | keine | alle Experten liegen auf derselben GPU |
| Token Dropping im Hauptmodell | keines | V2-Lite auf einem Gerät nutzt nur Expert-Balance; Qualitätsvergleich soll keine Tokens verwerfen |
| papergetreues MLA-Layout | ursprüngliche vier Attention-Zellen | Zweck ist die direkte Replikation des alten 2×2; das Layout wird separat in `paper_layout_followup` getestet |

Primärquellen: [DeepSeek-V2 Paper, besonders §2.2, §3.1.2 und Appendix B](https://arxiv.org/abs/2405.04434), [offizielle V2-Lite Modellimplementierung](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py), [offizielle V2-Lite Konfiguration](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json).

## 4. Vergleichbarkeitsinvarianten

Unverändert gegenüber den bisherigen Produktionsläufen:

- Datensatz: `HuggingFaceFW/fineweb-edu`, `sample-10BT`;
- dieselben lokalen `train.bin` (500M Tokens) und `val.bin` (5M Tokens);
- GPT-2-BPE, Kontext 512;
- 6 Layer, Hidden 512, 8 Heads;
- 15.300 Iterationen;
- Micro-Batch 16, Gradient Accumulation 4, effektive Batchgröße 64;
- 501.350.400 Trainingstokens pro Run;
- AdamW `(0.9, 0.95)`, Weight Decay 0.1;
- LR `6e-4 -> 6e-5`, 2.000 Warmup-Schritte, danach Cosine;
- bf16, kein `torch.compile`, Gradient Clipping 1.0;
- Seeds 42, 123, 456;
- Evaluation alle 500 Iterationen mit 600 Batches;
- finaler Run-Wert = Mittel der letzten fünf Validation-Evaluationen.

Verbesserung gegenüber dem alten 2×2: Datenfenster verwenden eigene, checkpointbare CPU-Generatoren. Für denselben Seed sehen alle vier Attention-Zellen exakt dieselben Trainings- und Validierungsfenster, unabhängig von der unterschiedlichen Modellinitialisierung.

`val_loss` und `train_loss` enthalten ausschließlich Next-Token-Cross-Entropy. Der Balance-Loss wird als `train_aux_loss`/`val_aux_loss` separat protokolliert; `optimization_loss = train_loss + train_aux_loss`. Damit bleiben die Qualitätsmetriken direkt vergleichbar.

## 5. Dateien und Ausgaben

- `model.py`: vier unveränderte Attention-Zellen plus skalierte DeepSeekMoE-Schichten;
- `train.py`: RNG-gepaartes Training, Resume, CE-/Aux-Logging und Routingdiagnostik;
- `test_correctness.py`: Gradienten, Layerplatzierung, Top-k, Balance-Formel, Compute und RNG;
- `test_reference.py`: unabhängige numerische Referenz für alle vier Attention-Modi;
- `run_experiments.ps1`: balancierter 12-Run-Plan;
- `analyze_results.py`: strenge Designprüfung, seed-gepaarte 2×2-Kontraste, historische Dense-Gegenüberstellung und Plots.

Die Analyse erzeugt:

- `plots/learning_curves.png`;
- `plots/final_val_loss.png`;
- `plots/heatmap_2x2.png`;
- `plots/router_load_heatmap.png`;
- `plots/router_balance.png`;
- `plots/dense_vs_moe.png`;
- CSVs mit Run-Endwerten, Zellmitteln, gepaarten Kontrasten, Architektur und Routerlast;
- `plots/RESULTS_GENERATED.md`.

## 6. Windows-/GPU-Prüfung

Die bestehende Umgebung liegt unter `experiments\.venv`. Vor dem Smoke-Test:

```powershell
cd C:\Pfad\zu\SeminarML\experiments
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Die Ausgabe muss `True` und die RTX 5070 enthalten. Falls die FineWeb-Dateien fehlen:

```powershell
.\.venv\Scripts\python.exe data\fineweb_edu\prepare.py
```

Der Smoke-Test nutzt alle vier Zellen, verkleinerte Dimensionen und echte CUDA-/bf16-Schritte. Erst wenn Tests und Smoke-Test vollständig durchlaufen, die zwölf Produktionsruns starten.

## 7. Erwartbare Laufzeiteigenschaft

Der aktive Matrixaufwand ist fast gleich zum Dense-FFN, die einfache PyTorch-Implementierung startet jedoch mehrere kleinere Experten-Kernels. Deshalb kann ein Run trotz ähnlicher FLOPs langsamer sein. `torch.compile` bleibt zur methodischen Vergleichbarkeit deaktiviert. Laufzeit ist keine Zielmetrik dieses Experiments; `elapsed_s` wird dennoch protokolliert.
