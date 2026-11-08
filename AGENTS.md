# Project instructions

Music Atlas compares acoustic features, spectrogram CNNs and pretrained embeddings
for music genre classification and audio similarity. Read the relevant task in `tasks/`
and the existing implementation before editing. Project documentation is in English.

- Use Python 3.12 and uv. Keep reusable logic in `src/music_atlas/`; CLI and FastAPI
  call the same functions. The frontend uses Next.js App Router and TypeScript.
- Write project documentation and all user-facing interface text in English. Keep screenshots
  and examples aligned with the English interface.
- Keep changes focused and readable. Reuse existing modules; introduce an abstraction
  only when the implementation needs it. Comments should explain non-obvious decisions.
- Follow the track, representation and artifact formats in `docs/EXPERIMENTS.md`.
  Fit preprocessing on train; keep artists, duplicates and fragments within one split.
- Store audio, caches, models and generated outputs outside Git. Resolve storage through
  `ATLAS_*` paths. Local and WCSS runs use the same code with different profiles.
- WCSS storage is Lustre: keep few large files there. FMA-full lives as verified tar shards
  (`docs/DEVELOPMENT.md`); read tracks from the shards, unpack only to node-local storage
  and write sharded outputs. Run downloads and preprocessing as CPU jobs, not on GPUs.
- Keep unavailable models explicit in API responses and UI. Distinguish measured results
  from examples and exploratory visualisations.
- For code changes, run Ruff and relevant Python tests; for frontend changes, run
  `npm run typecheck` and `npm run build` in `web/`.

Optional skills are stored in `.agents/skills/`: `ponytail` for implementation and review,
`frontend-design` for interface work, `uv-package-manager` for Python environments, and
`wcss-slurm` for WCSS jobs and storage. Their upstream revisions are recorded in
`.agents/skills.lock.json`. Read `wcss-slurm` before any WCSS work (jobs, storage,
transfers, quotas or access); generic SLURM advice is often wrong there.
