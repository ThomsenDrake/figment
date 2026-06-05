const sharedValidationInstructions = `
You are validating the Figment app scaffold in /Users/drake.thomsen/Documents/misc/figment.

This is a read-only validation job. Do not edit files, do not apply patches, do not commit, and do not clean the workspace.
Include uncommitted and untracked scaffold files in your review.

Primary frozen contracts to validate:
- Five Gradio tabs: Intake, Risk Check, Protocol Guidance, Navigator Output + Handoff, Trace.
- Primary model stack is NVIDIA Nemotron 3 Nano Omni.
- Base Nemotron 3 Nano plus Parakeet RNNT is stretch-only and must remain gated behind ALLOW_STRETCH_STACK.
- Audio-assisted intake only drafts editable fields; confirmed typed intake remains the source of truth before rules/navigation.
- Deterministic red-flag rules must not be downgraded by model output.
- Raw audio must not be retained in traces or published artifacts.

Return findings first, with severity and file/line references when possible. If you find no issues, say so clearly and list the verification you performed.
`;

export default workflow({
  name: "figment-scaffold-validation",
  version: "1",
  description:
    "Read-only validation sweep for the Figment Gradio scaffold, safety gates, protocol substrate, and ops/test harness.",
  maxConcurrency: 4,
  maxAgents: 4,
  phases: [
    {
      id: "validate",
      title: "Validate Scaffold",
      agents: [
        {
          id: "contracts-safety",
          title: "Contracts, safety gates, and model/audio boundaries",
          sandbox: "read-only",
          prompt: `${sharedValidationInstructions}

Focus areas:
- figment/config.py
- figment/schemas.py
- figment/validators.py
- figment/audio_intake.py
- figment/model_client.py
- figment/navigator.py
- traces/*.json

Check that Omni is the default primary model path, stretch stack is blocked unless explicitly allowed, audio drafts cannot flow into rules/navigation until confirmed, navigator output validation rejects downgrades/unsafe clinical language, and traces do not retain raw audio.`
        },
        {
          id: "protocol-runtime",
          title: "Deterministic rules, retrieval, SBAR, and protocol cards",
          sandbox: "read-only",
          prompt: `${sharedValidationInstructions}

Focus areas:
- data/protocol_cards/*.json
- figment/rules.py
- figment/retrieval.py
- figment/sbar.py
- scripts/build_fts.py

Check the 10-card protocol substrate, confirmed-intake rule gate, protocol_urgency floor behavior, retrieval fallback/index path, SBAR grounding, and any card/schema mismatches that could break the demo. Run lightweight local commands if helpful.`
        },
        {
          id: "gradio-ops-tests",
          title: "Gradio app, ops files, and tests",
          sandbox: "read-only",
          prompt: `${sharedValidationInstructions}

Focus areas:
- app.py
- tests/*.py
- requirements*.txt
- Dockerfile
- Makefile
- .env.example

Run the available test suite if possible. Check that the Gradio scaffold exposes the five frozen tabs, includes typed intake plus audio input, keeps disabled audio safe, has useful demo actions, and has coherent local/HF Space ops files.`
        },
        {
          id: "end-to-end-demo",
          title: "End-to-end demo path and residual risks",
          sandbox: "read-only",
          prompt: `${sharedValidationInstructions}

Focus areas:
- app.py
- figment/*.py
- data/protocol_cards/*.json
- traces/*.json
- docs/figment-workback-plan.md

Exercise or inspect an end-to-end canned demo path: intake confirmation, red-flag rules, protocol retrieval, navigator output, SBAR, and trace. Identify integration bugs, missing test coverage, or risks that would matter before committing/pushing the scaffold.`
        }
      ]
    }
  ]
});
