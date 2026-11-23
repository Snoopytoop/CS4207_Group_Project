# Declaration of AI use – Group 11

The module permits AI-generated code. With the agreement of the whole group, the code, figures, poster draft
and these documents were produced with an AI coding agent (**Claude Code**, Anthropic) working under the
group's direction, from a written brief that the group prepared (`AGENT_PROMPT.md` in the agent kit).

**What the agent did**
- Wrote the Python modules in `src/demand/`, the scripts in `scripts/` and the notebook source
  `demand_forecast.py`.
- Ran the notebook and benchmarks on Rockhill through Slurm.
- Generated the figures, the poster draft and the explainer / Q&A / script documents.

**What the group did** *(edit this section to describe your own contribution honestly)*
- Chose the task (demand forecasting) and wrote the brief, including the cleaning rules, split and the
  parallelism designs to use.
- Set up access to Rockhill (SSH key) and approved every command the agent ran on the shared server.
- Reviewed the outputs at each checkpoint: cleaning table, parallel = sequential checks, model vs baseline,
  speedups and non-determinism evidence.
- Studied `docs/EXPLAINER.md` and `docs/QA_PREP.md`, edited the poster, and rehearsed the presentation.
- *Add: anything you changed, re-ran, or checked yourselves.*

Every number on the poster comes from the notebook output `outputs/results.json`.
