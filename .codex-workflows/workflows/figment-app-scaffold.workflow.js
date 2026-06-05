const sharedInstructions = `
You are building the Figment app scaffold in /Users/drake.thomsen/Documents/misc/figment.

Read docs/figment-workback-plan.md and docs/prerequisites.md before editing.
The frozen primary architecture is NVIDIA Nemotron 3 Nano Omni. The split base Nemotron + Parakeet path is stretch-only and must remain gated config, not a default dependency.

You are not alone in the codebase. Other workers may be editing different files at the same time. Do not revert, overwrite, or clean up files outside your assigned write set. Do not commit. Do not edit docs/assets/mockups.

Build a minimal but real scaffold: importable modules, deterministic safety gates, canned/demo fallback, and enough tests/commands for the main coordinator to verify.
`;

export default workflow({
  name: "figment-app-scaffold",
  version: "1",
  description:
    "Fan out independent workers to scaffold the Figment Gradio app, deterministic protocol engine, model/audio adapters, and tests/ops.",
  maxConcurrency: 4,
  maxAgents: 4,
  phases: [
    {
      id: "scaffold",
      title: "Scaffold",
      agents: [
        {
          id: "core-contracts",
          title: "Core contracts and validators",
          sandbox: "workspace-write",
          prompt: `${sharedInstructions}

Owned write set:
- figment/__init__.py
- figment/config.py
- figment/schemas.py
- figment/trace.py
- figment/validators.py

Task:
Create the core package contracts. Include typed/dataclass or Pydantic-free structures suitable for a lightweight Gradio Space. Implement:
- config loading from env with MODEL_STACK, MODEL_BACKEND, AUDIO_BACKEND, ENABLE_AUDIO_INTAKE, ALLOW_STRETCH_STACK, LLAMA_BASE_URL, OMNI_ENDPOINT_URL, HF_MODEL_ID, FIGMENT_TRACE_DIR.
- legal/illegal config validation that keeps MODEL_STACK=omni_native as default and blocks base_nano_parakeet unless ALLOW_STRETCH_STACK=true.
- canonical constants for Omni IDs and stretch IDs from the workback plan.
- structured intake, red flag, audio draft, navigator output, and trace helper contracts.
- validators that reject invalid navigator output, empty source_cards, unknown card IDs, urgency below deterministic floor, uncited pathways, diagnosis/prescribing language, and unconfirmed audio intake.

Keep dependencies to the Python standard library. Return a summary of files changed and key APIs.`
        },
        {
          id: "protocol-engine",
          title: "Protocol cards, rules, retrieval, SBAR",
          sandbox: "workspace-write",
          prompt: `${sharedInstructions}

Owned write set:
- data/protocol_cards/*.json
- figment/rules.py
- figment/retrieval.py
- figment/sbar.py
- scripts/build_fts.py

Task:
Create the deterministic protocol substrate. Include the 10 protocol cards frozen in the workback plan with prototype/safety-boundary language. Implement:
- rules.py with deterministic red-flag checks over confirmed intake dictionaries and an urgency floor.
- retrieval.py with SQLite FTS/BM25 when an index exists and a simple in-memory fallback over JSON cards.
- sbar.py that renders a grounded SBAR note from validated navigator output.
- scripts/build_fts.py that builds a local SQLite FTS index from data/protocol_cards.

Keep code small and dependency-light. Do not touch app.py or tests. Return a summary of files changed and key APIs.`
        },
        {
          id: "model-audio-navigation",
          title: "Model, audio, prompt, navigation adapters",
          sandbox: "workspace-write",
          prompt: `${sharedInstructions}

Owned write set:
- figment/model_client.py
- figment/audio_intake.py
- figment/prompt_builder.py
- figment/navigator.py
- traces/demo_case_*.json
- data/demo_audio/.gitkeep

Task:
Create adapter scaffolding for model/audio/navigation without loading heavyweight model dependencies. Implement:
- model_client.py with hosted Omni, local llama.cpp OpenAI-compatible, and canned-response modes. Hosted/local can be simple HTTP JSON scaffolds with timeout handling.
- audio_intake.py with ENABLE_AUDIO_INTAKE=false-friendly behavior, canned transcript fallback, provider-neutral audio draft output, and stretch stack blocked unless allowed. Do not import NeMo.
- prompt_builder.py constrained prompt skeleton from the plan.
- navigator.py orchestration that assembles prompt, calls model_client, parses JSON or canned output, validates through validators if available, and emits trace data.
- at least three demo trace JSON files matching the canonical demo cases.
- data/demo_audio/.gitkeep only, no binary audio.

Do not touch protocol cards, app.py, requirements, or tests. Return a summary of files changed and key APIs.`
        },
        {
          id: "app-ops-tests",
          title: "Gradio app, ops, and tests",
          sandbox: "workspace-write",
          prompt: `${sharedInstructions}

Owned write set:
- app.py
- requirements.txt
- requirements-dev.txt
- Dockerfile
- Makefile
- .env.example
- tests/*.py

Task:
Create the runnable Gradio app scaffold and verification harness. Implement:
- app.py using Gradio Blocks with five tabs: Intake, Risk Check, Protocol Guidance, Navigator Output + Handoff, Trace.
- demo case buttons, typed intake, optional audio controls that work when disabled, confirm-intake gate, JSON trace display/download, local/offline and hosted-live status chips.
- requirements for Gradio app runtime, and dev requirements for pytest.
- Dockerfile and Makefile targets for install, test, run, build-fts.
- tests that exercise config legality, red-flag rules, validators, audio confirmation behavior, and an app smoke/import path. Tests may assume other workers' APIs but should be easy for coordinator to adjust after integration.

Do not touch module files outside app.py or docs/mockups. Return a summary of files changed and key APIs.`
        }
      ]
    }
  ]
});
