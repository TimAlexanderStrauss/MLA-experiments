# RTX-5070-Checkliste vor den 24 Produktionsruns

Diese Checkliste gilt für das Experiment in diesem Ordner. Arbeite die Abschnitte in der angegebenen Reihenfolge ab. Starte die 24 Produktionsruns erst, wenn die Pflichtprüfungen erfolgreich sind.

## 1. Umgebung öffnen

```powershell
cd C:\Pfad\zu\SeminarML\experiments\deepseek_mechanism_2x2x2
$Python = "..\.venv\Scripts\python.exe"
```

Prüfe Python, PyTorch, CUDA und die GPU:

```powershell
& $Python -c "import torch; print('torch=', torch.__version__); print('cuda_runtime=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('capability=', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); print('bf16=', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)"
```

Erfolgskriterien:

- `cuda_available=True`
- der GPU-Name enthält `RTX 5070`
- `bf16=True`
- PyTorch ist mindestens Version 2.6
- der Prozess meldet keine fehlende CUDA-Architektur

Bei einem Fehler sende die vollständige Ausgabe dieses Befehls.

## 2. Datensatz prüfen

Das neue Experiment muss dieselben Dateien wie die alten Experimente verwenden:

```powershell
Get-Item "..\data\fineweb_edu\train.bin", "..\data\fineweb_edu\val.bin" |
    Select-Object FullName, Length, LastWriteTime
```

Erfolgskriterien:

- beide Dateien existieren;
- der Pfad liegt unter `experiments\data\fineweb_edu`;
- `train.bin` ist deutlich größer als `val.bin`;
- es gibt keine separate Kopie des Datensatzes im neuen Experimentordner.

Optional kannst du Hashes für die spätere Reproduzierbarkeit speichern:

```powershell
Get-FileHash "..\data\fineweb_edu\train.bin", "..\data\fineweb_edu\val.bin" -Algorithm SHA256
```

## 3. CPU- und Referenztests ausführen

```powershell
.\run_tests.ps1
```

Erfolgskriterien:

- alle acht Modellzellen bestehen Forward und Backward;
- der batched MoE-Pfad stimmt mit der Expertenschleife überein;
- Router- und Balance-Loss-Tests bestehen;
- End-to-End-Kausalität, Loss-Abnahme und Resume-Äquivalenz bestehen;
- der kompilierte CPU-Kapazitätstest bricht wie erwartet ab;
- der Plan enthält genau 24 eindeutige Runs;
- die unabhängige RoPE-Referenz ist positionsabhängig;
- alle vier Attention-Referenztests melden eine maximale Abweichung unter `1e-4`;
- das Skript endet mit `All 2x2x2 experiment tests passed.`

Starte bei einem fehlgeschlagenen Test kein Training.

## 4. CUDA-Smoke-Test ausführen

```powershell
.\run_smoke_test.ps1
```

Dieser Test führt für alle acht Zellen einen bf16-Forward-/Backward-Durchlauf aus.
Danach provoziert er in einem getrennten Prozess einen Kapazitätsüberlauf mit
`torch.compile(mode="reduce-overhead")`.

Erfolgskriterien:

- alle acht Kombinationen aus Dense/MoE und den vier Attention-Modi erscheinen;
- alle Loss-Werte sind endlich;
- es gibt keinen CUDA-, bf16-, SDPA- oder Out-of-Memory-Fehler;
- `test_gpu_capacity.py` meldet den erwarteten Kapazitätsabbruch;
- das Skript endet mit `All eight CUDA smoke-test cells passed.`

Speichere die Ausgabe bei Bedarf:

```powershell
.\run_smoke_test.ps1 2>&1 | Tee-Object -FilePath "smoke_test_5070.log"
```

## 5. GPU-Leistungsprofil erzeugen

Entferne ein altes Profil nur dann, wenn es von einer anderen PyTorch-, CUDA- oder GPU-Umgebung stammt. Das Profil ist klein und kann vorher umbenannt werden.

Führe den vollständigen Benchmark aus:

```powershell
& $Python "benchmark_gpu.py" --full
```

Der Benchmark vergleicht:

- Micro-Batch 32, 16 und 8;
- Gradient Accumulation bei effektiver Batchgröße 64;
- batched und Loop-MoE-Dispatch;
- Ausführung mit und ohne `torch.compile`;
- Tokens pro Sekunde und Peak-VRAM.
- den echten Memmap-, `get_batch`- und Host-to-GPU-Datenpfad.

Prüfe danach das gewählte Profil:

```powershell
Get-Content "gpu_profile.json"
```

Erfolgskriterien:

- `device_name` bezeichnet die RTX 5070;
- `batch_size * grad_accum = 64`;
- `benchmark_tokens_per_second` ist größer als null;
- `benchmark_peak_vram_gib` liegt unter der 90-%-Grenze;
- `architecture.mla_d_c` ist `256`;
- der vollständige `architecture`-Block entspricht `run_experiments.py`;
- mindestens ein Profil hat den Status `ok`;
- das ausgewählte Profil ist das schnellste sichere Profil.

`compile` darf `true` oder `false` sein. Das Ergebnis ist von deiner PyTorch- und Triton-Version abhängig. Erzwinge `torch.compile` nicht, wenn der Benchmark die Eager-Ausführung als schneller auswählt.

Prüfe in `trials`, ob `batched` schneller als `loop` ist. Wenn `loop` schneller ist, sende die vollständige `gpu_profile.json`. Dann prüfen wir den batched Pfad auf dieser Umgebung erneut.

## 6. GPU-Auslastung beobachten

Öffne während des Benchmarks oder eines Trainingslaufs ein zweites PowerShell-Fenster:

```powershell
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 2
```

Beurteile nicht nur die GPU-Prozentzahl. Prüfe zuerst den Durchsatz in Tokens pro Sekunde.

Gute Zeichen:

- die GPU-Auslastung bleibt während der Rechenschritte überwiegend hoch;
- der Speicherverbrauch bleibt stabil;
- der Prozess zeigt keinen fortlaufenden Anstieg des reservierten VRAM;
- der batched Pfad erreicht mehr Tokens pro Sekunde als die alte Schleife.

Kurze Abfälle sind normal. Sie treten bei Evaluation, Checkpoints, Datenbereitstellung und Compiler-Start auf.

Wenn die Auslastung weiterhin ungefähr 75 % beträgt, notiere zusätzlich:

- das ausgewählte Profil;
- Tokens pro Sekunde für `batched` und `loop`;
- Micro-Batch und Gradient Accumulation;
- Peak-VRAM;
- `compile=true/false`;
- GPU-Takt, Leistungsaufnahme und Temperatur aus `nvidia-smi`.

## 7. Den 24-Run-Plan prüfen

Zeige den Plan an, ohne Training zu starten:

```powershell
& $Python "run_experiments.py" --dry_run
```

Erfolgskriterien:

- der Plan enthält 24 Startbefehle;
- beide Backbones erscheinen;
- alle vier Attention-Modi erscheinen;
- die Seeds 42, 123 und 456 erscheinen;
- alle Befehle verwenden dasselbe `gpu_profile.json`;
- alle Befehle verwenden `..\data\fineweb_edu`.

## 8. Einen vollständigen Pilot-Run ausführen

Führe vor den übrigen 23 Runs die größte und wichtigste Zelle aus:

```powershell
.\run_experiments.ps1 -Backbone moe -Attention mla_decoupled -Seed 42
```

Prüfe während des Runs:

- Loss bleibt endlich;
- Tokens pro Sekunde bleiben nach der Compiler-Warmup-Phase stabil;
- Peak-VRAM bleibt stabil;
- Router-Maximallast steigt nicht bis zu einem Kapazitätsfehler;
- Evaluation und Checkpoint-Schreiben funktionieren;
- ein unterbrochener Test kann aus `checkpoint.pt` fortgesetzt werden.

Nach erfolgreichem Abschluss müssen diese Dateien existieren:

```powershell
Get-ChildItem "results\moe_mla_decoupled_s42"
```

Erwartet werden:

- `config.json`
- `runtime.json`
- `metrics.csv`
- `routing.csv`
- `checkpoint.pt`
- `DONE`

Prüfe die letzten Metriken:

```powershell
Import-Csv "results\moe_mla_decoupled_s42\metrics.csv" |
    Select-Object -Last 5 |
    Format-Table iter, train_loss, val_loss, train_aux_loss, router_entropy, max_load_fraction, tokens_per_second, peak_vram_gib
```

## 9. Stop-Kriterien

Starte die übrigen Runs nicht, wenn einer dieser Fälle eintritt:

- ein Korrektheits- oder Referenztest schlägt fehl;
- der CUDA-Smoke-Test schlägt fehl;
- der Benchmark findet kein Profil unter der VRAM-Sicherheitsgrenze;
- das Profil stammt von einer anderen GPU-, PyTorch- oder CUDA-Version;
- Loss oder Gradienten werden `NaN` oder unendlich;
- der Router überschreitet die kompilierte Kapazität;
- VRAM oder Windows Commit steigen ohne Begrenzung;
- Resume ändert kontrollierte Parameter;
- der Pilot-Run erzeugt kein `DONE`-Flag.

Lösche bei einem Fehler keine Ergebnisordner und keine Checkpoints. Sende zuerst die Fehlermeldung und die zugehörige Konfiguration.

## 10. Alle Produktionsruns starten

Wenn alle vorherigen Prüfungen erfolgreich sind:

```powershell
.\run_experiments.ps1
```

Der bereits abgeschlossene Pilot-Run wird wegen seines `DONE`-Flags übersprungen.

Nach Abschluss aller Runs:

```powershell
.\analyze.ps1
```

Die Analyse muss alle 24 Runs und alle 24 `DONE`-Flags erkennen. Sie muss außerdem identische kontrollierte Hyperparameter sowie identische Daten-Seeds innerhalb jedes Seeds bestätigen.

## 11. Informationen für eine spätere Prüfung

Halte diese Dateien und Ausgaben bereit:

- `gpu_profile.json`
- `smoke_test_5070.log`, falls erstellt
- Ausgabe des PyTorch-/CUDA-Prüfbefehls aus Abschnitt 1
- Ausgabe des Datensatzbefehls aus Abschnitt 2
- die letzten fünf Zeilen von `metrics.csv` des Pilot-Runs
- relevante `nvidia-smi`-Werte
- vollständige Fehlermeldungen

Mit diesen Informationen können wir Leistung, VRAM, Compilerwahl und Routerkapazität prüfen, ohne die Produktionsruns neu zu starten.
