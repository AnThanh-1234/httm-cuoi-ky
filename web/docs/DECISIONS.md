# Decisions

- The app lives entirely under `web/`; the supplied `reference/` implementation is imported and not modified.
- `MOCK=1` returns deterministic, clearly labeled synthetic rankings so the interface can be exercised without a model download. This is not the SPEC's random-small-Qwen mock and is not benchmark evidence.
- In real mode, the backend loads the local Qwen backbone and any recognized adapter checkpoints from `MODELS_DIR`; unavailable adapters are reported as unavailable. Qwen may be downloaded on first use unless the Hugging Face cache is already populated.
- The retrieval gate remains false. No checkpoint evaluation is performed automatically at startup, and UI cannot override this flag.
- The example question and document are editable UI content. Kaggle figures are fetched from `/api/reference/kaggle`, not copied into frontend code.
- A session's adapter state remains in process memory and is rebuilt by ingesting the document; live tensor state is not serialized.
- Benchmark accepts caller-supplied items and computes top-1, MRR, gold logP, and bootstrap intervals. It is a basic benchmark runner, not the fixed 12-event held-out suite described in the SPEC.
- The backend gate check runs only on an explicitly designated evaluation request, at v2 checkpoints, with at least two seeds; production should replace this hook with a versioned, fixed held-out set before relying on the gate.
- CSV export is available for benchmark tables; session JSON export is implemented. PNG export, free-form generation, and Playwright smoke are not yet wired.
- Phase 0 Titans v2 is stored locally as `models/titans_v2.pt` (ignored by Git); its Kaggle metrics are in `docs/eval_v2.json` and are exposed separately as `titans_v2_factmix` from `/api/reference/kaggle`. It is a Titans-only run and does not replace the older comparison baseline.
- Titans checkpoint loading reads `cfg.titans_inner_lr_cap` and passes it to `NeuralMemory`; legacy checkpoints without the field retain the historical 1.0 default. The retrieval gate remains false because this Titans-only run does not include the paired window benchmark required by SPEC §8.
- Phase 0 Titans v2 is stored locally as `models/titans_v2.pt` (ignored by Git); its Kaggle metrics are in `docs/eval_v2.json` and are exposed separately as `titans_v2_factmix` from `/api/reference/kaggle`. It is a Titans-only run and does not replace the older comparison baseline.
- Titans checkpoint loading reads `cfg.titans_inner_lr_cap` and passes it to `NeuralMemory`; legacy checkpoints without the field retain the historical 1.0 default. The retrieval gate remains false because this Titans-only run does not include the paired window benchmark required by SPEC §8.
