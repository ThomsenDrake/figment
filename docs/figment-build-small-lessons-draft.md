# Building Figment For Build Small: What I Learned About Making Small Models Useful

I started Figment with a simple idea: what if a responder in a rural clinic, mobile unit, shelter, or disaster site had a protocol binder that could talk back?

Not an AI doctor. Not a system that decides whether someone should go home, receive treatment, or ignore a symptom. The version I wanted to build for the Hugging Face Build Small Hackathon was narrower than that: a protocol navigator that could take messy field intake, surface red flags, retrieve relevant protocol cards, ask for missing observations, and help draft a grounded SBAR-style handoff: Situation, Background, Assessment, and Recommendation.

The Build Small constraint made that idea more interesting. A lot of AI demos become persuasive by making the model bigger or the problem blurrier. Figment had to move the other way. The useful question was not "can a model answer medical questions?" It was "can a small-enough model do bounded, visible work inside a system that refuses to let it improvise?"

After the first few days, I thought the lesson was restraint. After the next few days of evals, failed prompts, corrected scoring, Modal runs, artifact publishing, and live Space work, the lesson got sharper:

Restraint is not just a safety pattern. It is an iteration engine.

Once the model's job is small enough to inspect, every failure can become a decision. Sometimes the decision is "train on this." Sometimes it is "fix the harness." Sometimes it is "the benchmark is wrong." Sometimes it is "the app already knows this deterministically, so stop asking the model to invent it."

That changed how I understand small-model product work.

## Audio Should Draft, Not Decide

One early design decision kept paying rent: audio intake should be a draft layer, not a decision layer.

In the field, speech is natural. A responder may not have the time, lighting, or hand freedom to fill out a perfect form. It is tempting to treat audio as magic: record the note, send it to a model, and let the app continue.

Figment does not do that.

Audio-derived text is treated as provisional. It can suggest fields like age, symptoms, vitals, allergies, medications, supplies, and free-text notes. But a human has to confirm or edit the intake before deterministic red-flag rules or navigator output run. Unconfirmed audio is not allowed to trigger final red flags, clear red flags, or drive the handoff.

That may sound like a small UX detail, but it is one of the most load-bearing safety choices in the app. ASR errors are not rare edge cases. A dropped negation or malformed field can change the meaning of a case. The safer product shape is not "voice in, answer out." It is "voice in, editable draft, confirmed facts, then navigation."

That shape carried through to the hosted Space. The live Parakeet route can draft fields from a committed demo WAV, but the result is still labeled unconfirmed. That label is not bureaucratic caution. It is the product contract.

## App Safety And Model Competence Are Different Numbers

This became the most important evaluation lesson of the project. Deterministic safety logic is not an embarrassing fallback. In Figment, it is the floor.

The app has deterministic red-flag rules for things like pediatric dehydration, respiratory distress, pregnancy danger signs, stroke signs, wound infection cues, and other prototype protocol-card categories. If those rules fire, the model cannot lower the urgency. The model can add useful structure around the case, but it does not get to reinterpret away the safety floor.

That changes the meaning of an eval. A safe final output does not necessarily mean the model performed well. It may mean deterministic rules caught the case, retrieval supplied the relevant cards, validators rejected unsafe output, and fallback kept the app inside the contract. A medical-adjacent prototype should not try to show that a model is safe by letting it be dangerous and hoping it behaves. It should make the model's job small enough that success and failure are both visible.

Early on, it would have been easy to report only final validation. The app could often produce a valid final navigator output because the scaffold was strong. But that would have hidden the real question for Build Small: was the model actually doing load-bearing work?

So I split the metrics.

In this post, competence means the configured model path did the load-bearing work for a case: it produced the required navigator behavior itself, or through an allowed focused model repair, without getting credit for deterministic fallback.

In the first 50-case eval against hosted Nemotron Omni, the hosted multimodal model route I used before the local 4B path, final validation passed `50/50`, but hosted model competence was only `28/50`. That distinction mattered. The app stayed inside its safety envelope, but the model was not carrying all of the work. Some cases needed deterministic fallback after hosted output failed validation or grounding checks.

After adding a more constrained prompt contract, field-level provenance, and focused repair, the hosted follow-up improved. Whole-output competence moved to `31/50`, full deterministic fallback dropped to `8/50`, and the field-level metric showed that the model was retaining most bounded fields while deterministic logic patched the rest.

That was the moment the eval started to feel honest. Instead of saying "the model passed" or "the app passed," I could say something more precise:

The application produced safe final outputs on the eval. The hosted model carried many bounded fields. Deterministic logic patched the rest. Full fallback still existed, and it was counted separately.

That distinction became even more important later. One local run looked perfect if I only counted final validation: v5 reached `150/150` final validation and `150/150` expected labels on the 150-case holdout. But the configured model was only competent on `2/150` cases, and deterministic patches were doing the heavy lifting.

That was not a victory lap. It was a smoke alarm.

If your app has fallback, validators, retrieval, and deterministic rules, do not collapse everything into one success number. A model-competence score and an app-safety score answer different questions.

The mechanism that made the split useful was field-level provenance. Instead of asking only "did the whole model response pass?", Figment started asking:

- Which fields came from the raw model?
- Which fields were repaired by a focused model call?
- Which fields were deterministically patched?
- Which cases required full fallback?

That changed the project. A model might select the right protocol pathway, ask useful missing-observation questions, and draft a reasonable checklist, while still failing one SBAR grounding rule. Field-level provenance lets the app keep validated parts and patch failed parts without pretending the whole output was model-generated.

It also makes the trace more useful. The Trace tab is not just a debugging feature; it is the project's honesty surface. It shows the intake, rules, retrieval, model output, validation, repair, fallback, provenance, and trace hashes. For a hackathon project, that might seem like a lot of plumbing. For a small-model project, it became the main way to show that the model was doing bounded work rather than being credited for deterministic scaffolding.

## Fine-Tuning Only Helped After The Eval Got Honest

The local 4B path was where the project got the most interesting and the most humbling.

The target was `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, served through a llama.cpp-compatible route after LoRA fine-tuning and GGUF conversion. The goal was not to make a general medical assistant. The goal was to teach a small local model the narrow Figment behavior: protocol-card discipline, red-flag preservation, missing-observation planning, safe handoff drafting, and schema-valid navigator JSON.

The first fine-tuning pilot was valuable because it demonstrated the full loop: generate teacher data, train on Modal, merge the adapter, convert to GGUF, serve locally, and run the eval harness against the local route. But the result was not a clean win. It landed at only `11/50` competence on the locked 50-case eval.

That failure was useful. It showed that training loss and JSON validity were not enough. The dataset had taught format more than judgment. It had too few examples for some failure modes. Some rows were not aligned tightly enough to the real harness. And the eval was punishing behaviors that looked safe in prose but violated the exact scorer or product contract.

The v2 dataset was a better answer. It used a stronger teacher model to generate synthetic, validated rows aligned to the actual Figment prompt and repair tasks, kept locked eval cases out of training, and added more repair rows and failure-class coverage. The v2 local model improved to `33/50` on the locked 50-case eval, with `50/50` final validation.

Then v3 changed the question again. Rather than only optimizing the locked 50-case eval, I created a 150-case field-workflow holdout for rural clinic intake, disaster triage, ASR-like confirmed text, low-resource constraints, radio handoff, SBAR usefulness, and source-card discipline. V3 reached `107/150` competence, with `93/150` raw local-model successes, `14/150` focused repair successes, `2/150` full fallbacks, and `148/150` final validation.

That sounded good, and in many ways it was. But the failure distribution mattered more than the headline score. The handoff layer was still weak. Radio handoff and SBAR usefulness were exactly where the model needed help.

That is where the next few days of work changed the project.

## The Eval Loop Got Teeth

The later Figment loop was less like one big training run and more like an eval-driven curriculum.

Here is the short version:

| Run | Competence | Raw success | Repair success | Expected labels | Final validation | Fallback | Deterministic patches | Lesson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v3 | 107/150 | 93/150 | 14/150 | 0/150 | 148/150 | 2 | 114 | Big field-workflow jump, but weak handoff behavior |
| v4 | 109/150 | 109/150 | 0/150 | 149/150 | 148/150 | 2 | 104 | Better raw output, still scaffold-heavy |
| v5 | 2/150 | 2/150 | 0/150 | 150/150 | 150/150 | 0 | 302 | The app passed; the model did not |
| v6 | 142/150 | 142/150 | 0/150 | 146/150 | 150/150 | 0 | 21 | Targeted replay and delta rows worked |
| v7 corrected | 148/150 | 148/150 | 0/150 | 147/150 | 150/150 | 0 | 3 | Failures became inspectable |
| v10 | 147/150 | 147/150 | 0/150 | 150/150 | 150/150 | 0 | 6 | Narrow misses resisted generic fixes |
| v14p | 146/150 | 146/150 | 0/150 | 150/150 | 150/150 | 0 | 8 | Raw path plateaued |
| v14p repair-union | 150/150 | 146/150 | 4/150 | 150/150 | 150/150 | 0 | 0 | Model repair closed the remaining cases |

Competence, raw success, repair success, expected labels, final validation, and fallback are case-level counts. Expected labels are the protocol/risk labels the holdout says the case should preserve. Deterministic patches are field-level counts: the number of output fields the deterministic layer overwrote or supplied.

The v14p repair-union row is the same v14p model with focused model repair enabled for the remaining cases.

The v3 expected-label number is not apples-to-apples with the later corrected runs. I include it as an early baseline, not as a clean leaderboard row.

The table is not a straight victory staircase. That is the point.

Some runs looked safer because the app was helping more. Some prompt probes made the model worse. Some data deltas moved the target metric, then plateaued. The useful thing was not that every version improved. The useful thing was that the eval made it possible to tell what kind of change had happened.

V5 is the run I keep coming back to because it would have been easy to misunderstand: the app passed and the model did not. That forced a decision. I could make the scaffold stronger and keep reporting final validation, or I could use v5 as evidence that the model-owned target needed more work. The second option made the rest of the project better.

V6 used targeted delta rows plus replay rows from earlier versions, training against the failure shape that v5 exposed. Competence moved from `2/150` to `142/150` while fallback stayed at zero. Later runs kept narrowing misses; some improved the exact metric, some did not, and a few looked nearly identical. That was frustrating, but it was also evidence that the eval had become specific enough to resist hand-wavy progress stories.

## Check The Target Before Training

The most valuable bug I found was not in the model. It was in the scoring and deterministic rule path.

The old holdout treated some negated phrases too bluntly. A sentence like "no chest pain reported" could still trigger a chest-pain-related signal because the matcher saw the words and missed the negation. That is the kind of bug that can make a benchmark look tougher while actually making it less faithful.

I did not mutate the original frozen holdout. Instead, I created a corrected scoring view with a manifest. It changed exactly six cases and kept the original and corrected hashes visible. That mattered because the point of an eval is trust. If the target moves, readers should be able to see how and why.

This became a new rule for the project: do not train your way around a bad benchmark. Fix the benchmark, leave a receipt, and rerun the model.

The same rule applied to prompts. Some failures involved missing required observation ownership, so I tested stricter prompt contracts that made required observation IDs more explicit and more mandatory. In theory, that should have helped. In practice, one mandatory-observation prompt probe made the run worse. That was a useful embarrassment: "prompt harder" was not a general solution.

The better loop became:

- identify the exact field or case family,
- decide whether the app, model, or scorer should own it,
- add targeted data only when the model really should own it,
- verify against the same holdout,
- keep raw, repair, patch, and fallback counts separate.

That loop is slower than prompt fiddling. It is also harder to fool.

## What V14p Actually Proves

The strongest current result is v14p repair-union on the corrected 150-case holdout. Repair-union means raw successes unioned with focused-repair successes: cases the first model pass got right plus cases a bounded model repair closed.

- `150/150` competence successes,
- `150/150` expected-label successes,
- `150/150` final-validation successes,
- zero deterministic patches,
- zero fallback uses,
- `146/150` raw configured-model successes,
- `4/150` cases resolved by focused model repair,
- no unsupported facts counted in the handoff metrics.

That is a strong result, but it has to be named precisely.

It does not mean every first-pass model response worked. It does not mean the app is ready for real-world deployment. It does not mean the synthetic holdout is a substitute for field testing with actual users.

It means the local 4B Figment system, with model-owned repair but without deterministic patching or fallback, can complete the corrected field-workflow holdout while preserving the prototype's safety and grounding contract.

That is a narrow claim. I like it because it is narrow enough to be useful.

## Make The Evidence Legible

Another thing I learned: evidence is not enough if the product surface does not make the evidence legible.

Figment started as a functional Gradio app. It had the pieces: intake, rules, retrieval, navigator output, trace. But it felt more like a harness than a field tool. The later UI work moved it to a custom Gradio Server surface with a "Field Kit Workbench" feel. The workflow became clearer without hiding the machinery that made the result trustworthy.

The hosted Space became part of the same lesson. It is one thing to have a repo, eval traces, and a local model story. It is another thing to make the public artifact actually run where judges will click it. On June 15, the Hugging Face Space was running on HF ZeroGPU at a pinned commit, with the published v14p BF16 archive preloaded and a ZeroGPU route wired into the app.

That deployment work had its own mini eval loop: runtime fixes, prompt compaction, route proof, then Parakeet ASR wiring. Parakeet CTC is the speech-to-text model used here for audio draft intake. A synthetic navigator call routed through HF ZeroGPU to the v14p archive, passed validation in about 42 seconds, and, importantly, still logged deterministic patches. A draft-audio call transcribed a committed demo WAV with Parakeet CTC in about 5 seconds and returned five draft fields, all marked unconfirmed with no raw audio stored.

That was a real upgrade to the demo story. In that June 15 check, the public Space could reach the tuned v14p archive through HF ZeroGPU and produce provisional audio-derived intake fields with Parakeet.

It is also not the same evidence as the corrected holdout result. The public route still showed deterministic patches in that synthetic navigation check. The ASR output is still unconfirmed draft intake. The local 4B route is the evidence for local-model operation; the hosted Space is evidence that the public demo route worked on HF infrastructure. The lesson is the same one Figment kept teaching me: deployment evidence, eval evidence, and product safety evidence are related, but they are not interchangeable.

The hackathon also changed how I think about artifact publishing. It is one thing to say "I trained a local model." It is another thing to publish model artifacts, dataset cards, configs, eval traces, Space commits, and schema-stable dataset viewers that let someone inspect the path. By the end of the later loop, the Hugging Face repos had public artifacts and dataset configs for v5 through v14p, with the v8-v14p corpora published through the same viewer-safe schema, and the Space itself had a live v14p ZeroGPU route plus Parakeet draft audio.

That matters because the interesting claim is rarely just the final score. The claim is the path: which rows were added, which cases were excluded, which eval was frozen, which scoring view was corrected, which artifacts were served, and which fallback paths were counted separately. Figment's edge is the depth of that evidence trail. The risk is that a deep evidence trail only helps if judges and users can understand it quickly.

## The Lesson I Am Taking From Build Small

Before this project, I would have described small-model product work mostly in terms of parameter count, latency, hardware, and model quality. Those still matter. But Figment made me think about "small" differently.

Small is also a design discipline.

It means starting with the boundary, not the model. It means narrowing the model's job until it can be checked. It means using deterministic rules where determinism is safer, making retrieval explicit, refusing to count fallback as model competence, and keeping traces detailed enough that a judge, user, or future builder can see what happened. It means letting the model contribute where language and prioritization matter, while keeping safety-critical floors outside the model's control.

The first version of this post ended with "make the next eval sharper." After v5 through v14p and the live Space work, I would say it a little differently:

Make the next failure smaller, clearer, and harder to hide.

That is the thing Figment taught me. Useful small-model apps are not small because they ask less ambitious questions. They are small because they are honest about where the model belongs, and because that honesty gives you a way to improve.
