# Project notes

## pipeline.md is the pipeline reference

[pipeline.md](pipeline.md) documents the whole dubbing pipeline (stages, models, key constants, files written,
run.json / chunks.json fields, endpoints).

- Read it before changing anything in `backend/`, `lipsync/` or the pipeline parts of `ui/index.html`.
- **Any change to the pipeline must update pipeline.md in the same change**: a stage added, removed or reordered;
  a model, service or prompt swapped; a key constant changed (thresholds, limits, speeds, batch sizes); a file
  written to the run folder; a run.json / chunks.json field; an endpoint or its body. Keep it accurate, not
  aspirational: if the code and pipeline.md disagree, fix whichever is wrong and say so.
