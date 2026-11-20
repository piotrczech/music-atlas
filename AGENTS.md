# Project instructions

Music Atlas compares acoustic features, spectrogram CNNs and pretrained embeddings
for music genre classification and audio similarity. Read the relevant task in `tasks/`,
the existing implementation and `notebooks/01-data-exploration.ipynb` (the data and split)
before editing.

- Use Python 3.12 and uv. Keep reusable logic in `src/music_atlas/`; the CLI, FastAPI and
  notebooks call the same functions. The frontend uses Next.js App Router and TypeScript.
- Write documentation and all user-facing text in English, with screenshots and examples
  matching the English interface.
- Keep changes focused and readable. Reuse existing modules; introduce an abstraction
  only when the implementation needs it. Comments explain non-obvious decisions.
- Follow the formats in `docs/EXPERIMENTS.md`: read the primary benchmark version from
  `[dataset] split` in `configs/` instead of hard-coding it, fit preprocessing on train, and
  keep artists, duplicates and fragments within one split.
- Keep audio, caches, models and generated outputs outside Git, resolved through `ATLAS_*`
  paths. Local and WCSS runs use the same code with different profiles.
- Develop and test every data or model step on the quick sample (`atlas quick-sample`, profile
  `quick`) first. Run the `full` profile on WCSS only after the quick run works end to end, and
  size the full job from the quick run's measured throughput (README, *Quick sample first*).
- Record data and model steps with `music_atlas.tracking.track()` (MLflow plus a run
  directory with `run.json`) and expose it as `--track` in the CLI; read the `atlas-tracking`
  skill before adding or comparing tracked runs.
- Keep unavailable models explicit in API responses and UI. Distinguish measured results
  from examples and exploratory visualisations.
- For code changes, run Ruff and the relevant Python tests; for frontend changes, run
  `npm run typecheck` and `npm run build` in `web/`.

## WCSS

Read the `wcss-slurm` skill before any WCSS work (jobs, storage, transfers, quotas or access);
generic SLURM advice is often wrong there.

- The clone lives in `$HOME`; source `~/.atlas-env.sh` (from
  `scripts/slurm/atlas-env.example.sh`) before any command. It points data, environments and
  the SLURM account at the music service; bill nothing to another service.
- Storage is Lustre: keep few large files. Read FMA-full tracks from its verified tar shards,
  unpack only to node-local storage and write sharded outputs.
- Run downloads and preprocessing as CPU jobs, not on GPUs. Use OOD notebooks for inspection only.

Optional skills in `.agents/skills/`: `ponytail` for implementation and review,
`frontend-design` for interface work, `uv-package-manager` for Python environments and
`wcss-slurm` for WCSS. Their upstream revisions are recorded in `.agents/skills.lock.json`.
`atlas-tracking` is the project's own skill (no upstream): MLflow conventions for runs, used
together with `wcss-slurm` for tracked jobs on WCSS.
