# Building Figment For Build Small: What I Learned About Making Small Models Useful

Draft status: rough public draft.

I started Figment with a simple idea: what if a responder in a rural clinic, mobile unit, shelter, or disaster site had a protocol binder that could talk back?

Not an AI doctor. Not a diagnosis engine. Not a system that decides whether someone should go home, get medication, or receive treatment. The version I wanted to build for the Hugging Face Build Small Hackathon was narrower than that: a protocol navigator that could take messy field intake, surface red flags, retrieve relevant protocol cards, ask for missing observations, and help draft a grounded SBAR handoff.

The Build Small constraint made that idea more interesting. A lot of AI demos become persuasive by making the model bigger or the problem blurrier. Figment had to move the other way. The useful question was not "can a model answer medical questions?" It was "can a small-enough model do bounded, visible work inside a system that refuses to let it improvise?"

That question changed almost everything about the project.

## Audio Should Draft, Not Decide

One of the first design decisions that stuck was that audio intake should be a draft layer, not a decision layer.

In the field, speech is natural. A responder may not have the time, lighting, or hand freedom to fill out a perfect form. It is tempting to treat audio as magic: record the note, send it to a model, and let the app continue.

Figment does not do that.

Audio-derived text is treated as provisional. It can suggest fields like age, symptoms, vitals, allergies, medications, supplies, and free-text notes. But a human has to confirm or edit the intake before deterministic red-flag rules or navigator output run. Unconfirmed audio is not allowed to trigger final red flags, clear red flags, or drive the handoff.

That may sound like a small UX detail, but it is actually one of the most load-bearing safety choices in the app. ASR errors are not rare edge cases. A dropped negation or malformed field can change the meaning of a case. I learned that the safe product shape is not "voice in, answer out." It is "voice in, editable draft, confirmed facts, then navigation."

The same lesson applied to the demo. I originally had audio upload and demo clips working, but the right primary workflow was live audio ingest, with upload as a backup. That made the demo closer to the actual setting Figment is meant for: a responder speaking into the tool, then correcting it before using the result.

## Deterministic Safety Rules Are The Floor

The second thing I learned is that deterministic safety logic should not be treated as an embarrassing fallback. In Figment, it is the floor.

The app has deterministic red-flag rules for things like pediatric dehydration, respiratory distress, pregnancy danger signs, stroke signs, wound infection cues, and other prototype protocol-card categories. If those rules fire, the model cannot lower the urgency. The model can add useful structure around the case, but it does not get to reinterpret away the safety floor.

That sounds obvious when written down. In practice, it changes how you evaluate the model. A safe final output does not necessarily mean the model performed well. It may mean the deterministic layer caught the case, retrieval supplied the relevant cards, validators rejected unsafe output, and fallback kept the app inside the contract.

For a while, that made the project feel less impressive. Then I realized it made the project more honest.

A medical-adjacent prototype should not try to prove that a model is safe by letting it be dangerous and hoping it behaves. It should make the model's job small enough that success and failure are both visible. Figment's deterministic rules, protocol cards, validators, traces, and fallback paths are not there because I do not believe in small models. They are there because I want to know exactly where the model helped and exactly where it did not.

## App Safety And Model Competence Are Different Numbers

This became the most important evaluation lesson of the project.

Early on, it would have been easy to report only final validation. The app could often produce a valid final navigator output because deterministic fallback was strong. But that would have hidden the real question for Build Small: was the model actually doing load-bearing work?

So I split the metrics.

In the first 50-case hosted Omni eval, final validation passed `50/50`, but hosted model competence was only `28/50`. That distinction mattered. The app stayed inside its safety envelope, but the model was not carrying all of the work. Some cases needed deterministic fallback after the hosted output failed validation or grounding checks.

After adding a more constrained prompt contract, field-level provenance, and focused repair, the hosted follow-up improved. Whole-output competence moved to `31/50`, full deterministic fallback dropped to `8/50`, and the field-level metric showed `480/650` model-retained fields, with `170/650` deterministic patches.

That was the moment the eval started to feel honest. Instead of saying "the model passed" or "the app passed," I could say something more precise:

The application produced safe final outputs on the eval. The hosted model carried many bounded fields. Deterministic logic patched the rest. Full fallback still existed, and it was counted separately.

That distinction is probably the single most important thing I would recommend to other builders in this space. If your app has fallback, validators, retrieval, and deterministic rules, do not collapse everything into one success number. A model-competence score and an app-safety score answer different questions.

## Field-Level Provenance Changed My Relationship With Fallback

The first version of Figment treated model output mostly as all-or-nothing JSON. If one important field failed validation, the app could fall back to deterministic output. That was safe, but it also threw away useful model work.

The better pattern was field-level provenance.

Instead of asking "did the whole model response pass?", Figment started asking:

- Which fields came from the raw model?
- Which fields were repaired by a focused model call?
- Which fields were deterministically patched?
- Which cases required full fallback?

That changed the project. A model might select the right protocol pathway, ask useful missing-observation questions, and draft a reasonable checklist, while still failing one SBAR grounding rule. Field-level provenance lets the app keep the validated parts and patch the failed parts without pretending the whole output was model-generated.

It also makes the trace more useful. The Trace tab is not just a debugging feature; it is the project's honesty surface. It shows input, rules, retrieval, prompt context, model output, validation, repair, fallback, and provenance. For a hackathon project, that might seem like a lot of plumbing. But for a small-model project, it became the main way to show that the model was doing bounded work rather than being credited for deterministic scaffolding.

I came away thinking that fallback is not one thing. There is a big difference between:

- the model succeeded raw,
- the model succeeded after focused repair,
- the model contributed some fields,
- the model failed and deterministic fallback produced the result.

Those distinctions matter if you want to make credible claims about small models.

## Fine-Tuning Only Helped After The Eval Got More Honest

The local 4B path was where the project got the most interesting and the most humbling.

The target was `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, served locally through a llama.cpp-compatible route after fine-tuning and GGUF conversion. The goal was not to make a general medical assistant. The goal was to teach a small local model the narrow Figment behavior: protocol-card discipline, red-flag preservation, missing-observation planning, safe handoff drafting, and schema-valid navigator JSON.

The first fine-tuning pilot was valuable because it proved the full loop: generate teacher data, train on Modal, merge the adapter, convert to GGUF, serve locally, and run the eval harness without cloud inference. But the result was not a clean win. The pilot made the model better at shape and field retention, but it regressed competence to `11/50` on the locked 50-case eval.

That failure was useful. It showed that training loss and JSON validity were not enough. The dataset had taught format more than judgment. It had too few examples for some failure modes. Some rows were not aligned tightly enough to the real harness. And the eval was punishing behaviors that looked safe in prose but violated the exact scorer or product contract.

The v2 dataset was a better answer. It used a stronger teacher model to generate synthetic, validated rows aligned to the actual Figment prompt and repair tasks. It kept locked eval cases out of training. It added more repair rows and failure-class coverage. The v2 local model improved to `33/50` on the locked 50-case eval, with `50/50` final validation.

Then v3 changed the question again.

Rather than only optimizing the locked 50-case eval, I created a 150-case field-workflow holdout. That holdout asked whether Figment helped the real workflow: rural clinic intake, disaster triage, ASR-like confirmed text, low-resource constraints, radio handoff, SBAR usefulness, and source-card discipline.

The Modal v3 training run completed cleanly: `700/700` steps, final `eval_loss=0.04357146`, final `train_loss=0.60960097`, with adapter artifacts present in `figment-checkpoints:/figment_sft_v3/figment-sft-v3-lora`. But the important proof still came after training, when the model was merged, converted, served locally, and evaluated.

On the 150-case field-workflow holdout, v3 reached `107/150` competence, with `93/150` raw local-model successes, `14/150` focused repair successes, `2/150` full fallbacks, and `148/150` final validation.

That sounds good, and in many ways it was. But the failure distribution mattered more than the headline score.

## What Still Is Not Good Enough

V3 was strong on many of the safety and protocol-navigation surfaces. It preserved urgency floors. It matched red flags. It usually kept source cards and candidate pathways in shape. It had very low full fallback.

But it was bad at the handoff layer.

The field-workflow holdout exposed that clearly:

- `REFERRAL-SBAR-v1`: `0/27` competence
- `radio_handoff`: `0/16` competence
- `sbar_handoff_usefulness`: `0/10` competence

This was exactly the kind of failure a broader eval might hide. The model could produce valid JSON. It could be safe. It could cite cards. But it was not yet reliably producing the compact, grounded, operationally useful SBAR/radio handoff support that the field workflow actually needs.

That changed how I thought about the next step. The answer is not "train bigger" or "train more" in a generic way. The answer is to separate what the model should own from what the harness should own.

Some expected cues in the eval were things the app knows deterministically: retrieved card IDs, deterministic rule results, validator status, confirmed intake status, manual correction status for audio-derived fields. The model should not have to memorize or restate those as if they are missing observations. The app should surface them as deterministic evidence badges.

The model-owned part is different: concise handoff language, relevant background, observation-only assessment, a specific request, source-card discipline, and the next few observations that would actually help a responder move the case forward.

That is the next frontier for Figment.

## What I Would Build Next

If I had another iteration, I would not start by launching another broad training run.

First I would fix the eval shape. I would split expected cues into model-owned observations, handoff-owned cues, and harness-owned evidence. The app should deterministically expose metadata like validation status and retrieved card IDs. The model should be evaluated on the text it is actually responsible for.

Second, I would make SBAR and radio handoff first-class eval surfaces. Instead of letting those failures hide under a generic missing-observation score, I would measure:

- situation present,
- relevant background present,
- assessment is observation-only,
- request is specific,
- source cards are cited,
- red flags remain visible,
- unsupported facts are absent,
- handoff is brief enough to use.

Third, I would train a focused v4 dataset only after those scaffolding changes. The dataset should be narrow: radio handoff, SBAR usefulness, source-card discipline, low-resource constraints, high-value next observations, and focused repair rows from v3 safe-but-weak outputs. It should include enough replay rows to preserve the v2/v3 safety and schema gains, but not become another generic "more rows" pass.

The lesson from v1, v2, and v3 is that fine-tuning only helps when the target is clear. If the eval asks the model to satisfy app-owned metadata cues, the training run may learn the wrong thing. If the eval measures the field workflow, the training run has a chance to improve the product.

## The Lesson I Am Taking From Build Small

Before this project, I would have described small-model product work mostly in terms of parameter count, latency, hardware, and model quality. Those still matter. But Figment made me think about "small" differently.

Small is also a design discipline.

It means narrowing the model's job until it can be checked. It means using deterministic rules where determinism is safer. It means making retrieval explicit. It means refusing to count fallback as model competence. It means keeping traces detailed enough that a judge, user, or future builder can see what happened. It means letting the model contribute where language and prioritization matter, while keeping safety-critical floors outside the model's control.

In that sense, Figment is not just a prototype protocol navigator. It is an argument for a way of building with small models:

Give the model a real job, but make the job bounded.

Measure whether it did that job, not whether the app survived around it.

Keep the safety rails visible.

And when the model fails, do not hide the failure. Use it to make the next eval sharper.

That is the thing I learned building Figment for Build Small. Useful small-model apps are not small because they ask less ambitious questions. They are small because they are honest about where the model belongs.
