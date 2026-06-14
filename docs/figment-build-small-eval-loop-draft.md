# The Eval Loop That Made Figment Better

Draft status: rough technical follow-up draft for builders.

The first Figment retrospective was about restraint: keep the model inside a narrow job, make rules deterministic where safety needs determinism, and separate app safety from model competence.

Two days later, I would add a second lesson:

Once restraint is real, the eval loop gets teeth.

Figment improved because the harness got specific enough to make vague progress impossible. A run could no longer hide behind valid JSON. It could no longer borrow credit from deterministic fallback. It had to show which fields came from the raw model, which fields came from focused model repair, which fields came from deterministic patches, and which cases required full fallback.

That made the later training loop uncomfortable in exactly the right way.

## The Dangerous Shortcut Was Counting The App

The easiest number to report is final validation.

For Figment, that number answers a real question: did the app produce a schema-valid, grounded, safety-preserving navigator output? It matters. A user does not care whether a malformed model response had good intentions.

But final validation is not the Build Small question. The Build Small question is whether the small model did useful bounded work.

That is why v5 was so useful and so humbling. It reached `150/150` final validation and `150/150` expected labels on the 150-case holdout. If I had stopped there, I could have told a flattering story. But the configured model was competent on only `2/150` cases. Deterministic patches were doing most of the work.

That run forced the metric split:

- final validation: did the app stay inside the contract?
- expected labels: did the final output preserve the case-level safety targets?
- raw model competence: did the configured model produce a competent response before repair?
- repair success: did a focused model repair fix a bounded field failure?
- deterministic patches: did code patch over model output?
- fallback: did the app abandon the model path and use deterministic output?

Those numbers have different meanings. Combining them makes the project look smoother and teaches you less.

## A Short Version History

The later Figment loop looked roughly like this:

| Run | Competence | Raw success | Repair success | Expected labels | Final validation | Fallback | Deterministic patches | Lesson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v3 | 107/150 | 93/150 | 14/150 | 0/150 | 148/150 | 2 | 114 | Big jump, but weak handoff behavior |
| v4 | 109/150 | 109/150 | 0/150 | 149/150 | 148/150 | 2 | 104 | Better raw output, still scaffold-heavy |
| v5 | 2/150 | 2/150 | 0/150 | 150/150 | 150/150 | 0 | 302 | The app passed; the model did not |
| v6 | 142/150 | 142/150 | 0/150 | 146/150 | 150/150 | 0 | 21 | Targeted replay and delta rows worked |
| v7 corrected | 148/150 | 148/150 | 0/150 | 147/150 | 150/150 | 0 | 3 | Failures became inspectable |
| v10 | 147/150 | 147/150 | 0/150 | 150/150 | 150/150 | 0 | 6 | Narrow misses resisted generic fixes |
| v14p | 146/150 | 146/150 | 0/150 | 150/150 | 150/150 | 0 | 8 | Raw path plateaued |
| v14p repair-union | 150/150 | 146/150 | 4/150 | 150/150 | 150/150 | 0 | 0 | Model repair closed the remaining cases |

The v3 expected-label number is not apples-to-apples with the later corrected runs. I include it as an early baseline, not as a clean leaderboard row.

The table is not a straight victory staircase. That is the point.

Some runs looked safer because the app was helping more. Some prompt probes made the model worse. Some data deltas moved the target metric, then plateaued. The useful thing was not that every version improved. The useful thing was that the eval made it possible to tell what kind of change had happened.

## V5 Was The Run That Prevented A Bad Blog Post

V5 is the run I keep coming back to because it would have been easy to misunderstand.

At the app level, v5 looked great: no fallback, all final outputs valid, all expected labels preserved. Under the hood, it was exactly the wrong kind of success. Deterministic patches were compensating for model failures across the output.

That forced a decision. I could make the scaffold stronger and keep reporting final validation, or I could use v5 as evidence that the model-owned target needed more work.

The second option made the rest of the project better.

V6 used a targeted corpus: `1430` new delta rows plus `570` replay rows from earlier versions. It did not throw away the previous gains. It trained against the failure shape that v5 exposed. The result moved competence from `2/150` to `142/150` while keeping fallback at zero.

V7 pushed further with `2800` total rows, including `800` new delta rows and replay from v3 through v6. It reached `148/150` competence. More importantly, it reduced the remaining problem to a small number of cases that could be read, argued with, and turned into concrete next actions.

## The Best Fix Was Sometimes To Fix The Eval

One of those concrete actions was not a training run.

The holdout had a negation problem. Some logic treated phrases like "no chest pain reported" too much like positive chest pain evidence. That kind of bug is subtle because it can make an eval look stricter while making it less true.

I fixed the rule behavior and created a corrected scoring view instead of silently rewriting the frozen holdout. The manifest says exactly what changed: six cases, with original and corrected hashes preserved.

That was a turning point in how I thought about evaluation hygiene. A benchmark is not sacred because it is frozen. It is useful because it is inspectable, stable, and honest. If it is wrong, the answer is not to train a model to satisfy the wrong target. The answer is to create a corrected view with a receipt.

## Prompt Contracts Were Not Free

Another failure class involved required observation ownership, especially around postpartum fever cases. The tempting fix was to push more of the desired behavior into the prompt: make required observation IDs more explicit, make the policy more mandatory, make the model promise harder.

That did not reliably help.

One stricter prompt probe made the corrected holdout result worse. Other prompt variants shifted failure patterns without solving the underlying ownership problem. The lesson was not "prompts do not matter." Figment's earlier gains depended on prompt shape. The lesson was that prompt contracts are capacity tradeoffs. In a narrow 4B setup, extra instructions can compete with the actual task.

The better loop was:

- identify the exact field or case family,
- decide whether the app, model, or scorer should own it,
- add targeted data only when the model really should own it,
- verify against the same holdout,
- keep raw, repair, patch, and fallback counts separate.

That loop is slower than prompt fiddling. It is also harder to fool.

## What V14p Actually Proves

The strongest current result is v14p repair-union on the corrected 150-case holdout:

- `150/150` competence successes,
- `150/150` expected-label successes,
- `150/150` final-validation successes,
- zero deterministic patches,
- zero fallback uses,
- `146/150` raw configured-model successes,
- `4/150` cases resolved by focused model repair,
- `1942` fields from raw model output,
- `8` fields from model repair,
- no unsupported facts counted in the handoff metrics.

That is a strong result, but it has to be named precisely.

It does not mean the raw model solved every case on the first pass. It does not mean the app is validated for real-world deployment. It does not mean the synthetic holdout is a substitute for field testing with actual users.

It means the local 4B Figment system, with model-owned repair but without deterministic patching or fallback, can complete the corrected field-workflow holdout while preserving the prototype's safety and grounding contract.

That is a narrow claim. I like it because it is narrow enough to be useful.

## Publishing Receipts Changed The Work

This loop also changed how I thought about publishing.

If the only artifact is a demo, the audience has to trust your narrative. If the artifacts are public, the narrative becomes inspectable. For Figment, that meant publishing and verifying model artifacts, dataset cards, dataset configs, viewer schemas, and eval traces across versions.

The Hugging Face dataset configs make the curriculum visible: v6 at `1800/200`, v7 at `2520/280`, then v8-v14p as targeted follow-on corpora with stable 47-column viewer schema. The model repo carries the versioned artifacts. The trace directories preserve how each run behaved.

That matters because the most interesting part of this project is not the final number. It is the audit trail from failure to targeted data to rerun to new failure.

## What I Would Do Next

The next version of this work should not be another blind training run.

I would first make the demo story faster to understand. The competitor scan made that obvious: projects like Dental SOAP are very clear at first glance. Figment has deeper receipts, but the demo has to communicate the Backyard scenario quickly.

I would also add more user-facing evaluation around handoff usefulness. The current holdout measures radio/SBAR behavior, source support, unsupported facts, and validation, but real responders would expose different friction: wording, order, brevity, confidence, and whether the next observations are actually actionable under low-resource constraints.

Finally, I would keep the model/app boundary visible. The best future Figment is not the one that hides all scaffolding and pretends the model is autonomous. It is the one that makes each responsibility legible: human confirmation, deterministic red-flag floors, retrieval, model reasoning within a narrow contract, model repair, validation, and trace.

That is the main thing this eval loop taught me. Small models get better when the system around them is honest enough to make their failures specific.
