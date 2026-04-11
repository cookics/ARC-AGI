# Nemotron ARC-1 Runner

Standalone OpenRouter runner for ARC-AGI v1 training tasks.

Default model: `nvidia/nemotron-3-super-120b-a12b:free`

Usage:

```powershell
python nemotronData/openrouter_arc1_nemotron.py --task-id 007bbfb7
python nemotronData/openrouter_arc1_nemotron.py --limit 400 --workers 1
```

Artifacts are written under `nemotronData/runs/`.
