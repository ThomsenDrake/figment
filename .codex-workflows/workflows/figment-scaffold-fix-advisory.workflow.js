const sharedFixInstructions = `
You are advising on fixes for the Figment app scaffold in /Users/drake.thomsen/Documents/misc/figment.

This is a read-only advisory job. Do not edit files, do not apply patches, do not commit, and do not clean the workspace.
Include uncommitted and untracked scaffold files in your review.

Frozen contracts:
- Primary model stack is NVIDIA Nemotron 3 Nano Omni.
- Base Nemotron 3 Nano plus Parakeet RNNT is stretch-only and must remain gated behind ALLOW_STRETCH_STACK.
- Audio-assisted intake only drafts editable fields; typed/edited intake must be confirmed before rules/navigation.
- Deterministic red-flag rules must not be downgraded by model output.
- Raw audio and raw audio identifiers must not be retained in traces or published artifacts.

Return actionable implementation guidance with file/function targets, edge cases, and tests to add or update.
`;

export default workflow({
  name: "figment-scaffold-fix-advisory",
  version: "1",
  description:
    "Read-only implementation guidance for fixing Figment scaffold validation findings.",
  maxConcurrency: 4,
  maxAgents: 4,
  phases: [
    {
      id: "advise",
      title: "Advise Fixes",
      agents: [
        {
          id: "navigator-safety",
          title: "Navigator validation, urgency floor, and trace scrubbing",
          sandbox: "read-only",
          prompt: `${sharedFixInstructions}

Focus on figment/navigator.py, figment/validators.py, figment/trace.py, and tests.
Advise how to fail closed or sanitize navigator output when validation fails, preserve deterministic urgency floors, remove forbidden clinical language, and prevent raw audio-like payloads from trace output.`
        },
        {
          id: "rules-negation",
          title: "Deterministic rule false positives and negative cases",
          sandbox: "read-only",
          prompt: `${sharedFixInstructions}

Focus on figment/rules.py and tests.
Advise how to stop pregnancy/fever rules from matching field labels or negated text such as not_applicable, not pregnant, and no fever while preserving red-flag sensitivity.`
        },
        {
          id: "audio-ui",
          title: "Editable audio drafts and UI confirmation contract",
          sandbox: "read-only",
          prompt: `${sharedFixInstructions}

Focus on app.py, figment/audio_intake.py, figment/config.py, and tests.
Advise how to make audio suggestions editable before confirmation, avoid auto-accepting unreviewed fields, label canned and uploaded-audio behavior honestly, and keep raw audio out of traces.`
        },
        {
          id: "ops-artifacts",
          title: "Configured model routing, README, and trace artifacts",
          sandbox: "read-only",
          prompt: `${sharedFixInstructions}

Focus on app.py, README.md, traces/*.json, .env.example, and tests.
Advise how to pass the configured model backend through the Gradio flow, refresh demo traces, and align documentation with Omni primary plus stretch-only split stack.`
        }
      ]
    }
  ]
});
