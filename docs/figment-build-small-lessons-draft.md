# Building Figment For Build Small: What I Learned About Making Small Models Useful

Draft status: rough public draft, refreshed June 13, 2026 after the v5-v14p training/eval loop.

I started Figment with a simple idea: what if a responder in a rural clinic, mobile unit, shelter, or disaster site had a protocol binder that could talk back?

Not an AI doctor. Not a system that decides whether someone should go home, receive treatment, or ignore a symptom. The version I wanted to build for the Hugging Face Build Small Hackathon was narrower than that: a protocol navigator that could take messy field intake, surface red flags, retrieve relevant protocol cards, ask for missing observations, and help draft a grounded SBAR handoff.

The Build Small constraint made that idea more interesting. A lot of AI demos become persuasive by making the model bigger or the problem blurrier. Figment had to move the other way. The useful question was not "can a model answer medical questions?" It was "can a small-enough model do bounded, visible work inside a system that refuses to let it improvise?"

After the first few days, I thought the lesson was restraint. After the next two days of evals, failed prompts, corrected scoring, H100 runs, and public artifact publishing, the lesson got sharper:

Restraint is not just a safety pattern. It is an iteration engine.

Once the model's job is small enough to inspect, every failure can become a decision. Sometimes the decision is "train on this." Sometimes it is "fix the harness." Sometimes it is "the benchmark is wrong." Sometimes it is "the app already knows this deterministically, so stop asking the model to invent it."

That changed how I understand small-model product work.

## Audio Should Draft, Not Decide

One of the first design decisions that stuck was that audio intake should be a draft layer, not a decision layer.

In the field, speech is natural. A responder may not have the time, lighting, or hand freedom to fill out a perfect form. It is tempting to treat audio as magic: record the note, send it to a model, and let the app continue.

Figment does not do that.

Audio-derived text is treated as provisional. It can suggest fields like age, symptoms, vitals, allergies, medications, supplies, and free-text notes. But a human has to confirm or edit the intake before deterministic red-flag rules or navigator output run. Unconfirmed audio is not allowed to trigger final red flags, clear red flags, or drive the handoff.

That may sound like a small UX detail, but it is one of the most load-bearing safety choices in the app. ASR errors are not rare edge cases. A dropped negation or malformed field can change the meaning of a case. The safer product shape is not "voice in, answer out." It is "voice in, editable draft, confirmed facts, then navigation."

The same lesson applied to the demo. I originally had audio upload and demo clips working, but the better primary workflow was live audio ingest, with upload as a backup. That made the demo closer to the actual setting Figment is meant for: a responder speaking into the tool, then correcting the draft before using it.

## Deterministic Safety Rules Are The Floor

The second thing I learned is that deterministic safety logic should not be treated as an embarrassing fallback. In Figment, it is the floor.

The app has deterministic red-flag rules for things like pediatric dehydration, respiratory distress, pregnancy danger signs, stroke signs, wound infection cues, and other prototype protocol-card categories. If those rules fire, the model cannot lower the urgency. The model can add useful structure around the case, but it does not get to reinterpret away the safety floor.

That sounds obvious when written down. In practice, it changes how you evaluate the model. A safe final output does not necessarily mean the model performed well. It may mean the deterministic layer caught the case, retrieval supplied the relevant cards, validators rejected unsafe output, and fallback kept the app inside the contract.

For a while, that made the project feel less impressive. Then I realized it made the project more honest.

A medical-adjacent prototype should not try to prove that a model is safe by letting it be dangerous and hoping it behaves. It should make the model's job small enough that success and failure are both visible. Figment's rules, protocol cards, validators, traces, and fallback paths are not there because I do not believe in small models. They are there because I want to know exactly where the model helped and exactly where it did not.

## App Safety And Model Competence Are Different Numbers

This became the most important evaluation lesson of the project.

Early on, it would have been easy to report only final validation. The app could often produce a valid final navigator output because deterministic fallback was strong. But that would have hidden the real question for Build Small: was the model actually doing load-bearing work?

So I split the metrics.

In the first 50-case hosted Omni eval, final validation passed `50/50`, but hosted model competence was only `28/50`. That distinction mattered. The app stayed inside its safety envelope, but the model was not carrying all of the work. Some cases needed deterministic fallback after hosted output failed validation or grounding checks.

After adding a more constrained prompt contract, field-level provenance, and focused repair, the hosted follow-up improved. Whole-output competence moved to `31/50`, full deterministic fallback dropped to `8/50`, and the field-level metric showed `480/650` model-retained fields, with `170/650` deterministic patches.

That was the moment the eval started to feel honest. Instead of saying "the model passed" or "the app passed," I could say something more precise:

The application produced safe final outputs on the eval. The hosted model carried many bounded fields. Deterministic logic patched the rest. Full fallback still existed, and it was counted separately.

That distinction became even more important later. One local run looked perfect if I only counted final validation: v5 reached `150/150` final validation and `150/150` expected labels on the 150-case holdout. But the configured model was only competent on `2/150` cases, and deterministic patches were doing the heavy lifting.

That was not a victory lap. It was a smoke alarm.

If your app has fallback, validators, retrieval, and deterministic rules, do not collapse everything into one success number. A model-competence score and an app-safety score answer different questions.

## Field-Level Provenance Changed My Relationship With Fallback

The first version of Figment treated model output mostly as all-or-nothing JSON. If one important field failed validation, the app could fall back to deterministic output. That was safe, but it also threw away useful model work.

The better pattern was field-level provenance.

Instead of asking "did the whole model response pass?", Figment started asking:

- Which fields came from the raw model?
- Which fields were repaired by a focused model call?
- Which fields were deterministically patched?
- Which cases required full fallback?

That changed the project. A model might select the right protocol pathway, ask useful missing-observation questions, and draft a reasonable checklist, while still failing one SBAR grounding rule. Field-level provenance lets the app keep the validated parts and patch the failed parts without pretending the whole output was model-generated.

It also makes the trace more useful. The Trace tab is not just a debugging feature; it is the project's honesty surface. It shows input, rules, retrieval, prompt context, model output, validation, repair, fallback, and provenance. For a hackathon project, that might seem like a lot of plumbing. For a small-model project, it became the main way to show that the model was doing bounded work rather than being credited for deterministic scaffolding.

Fallback is not one thing. There is a big difference between:

- the model succeeded raw,
- the model succeeded after focused repair,
- the model contributed some fields,
- the model failed and deterministic fallback produced the result.

Those distinctions matter if you want to make credible claims about small models.

## Fine-Tuning Only Helped After The Eval Got Honest

The local 4B path was where the project got the most interesting and the most humbling.

The target was `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, served through a llama.cpp-compatible route after LoRA fine-tuning and GGUF conversion. The goal was not to make a general medical assistant. The goal was to teach a small local model the narrow Figment behavior: protocol-card discipline, red-flag preservation, missing-observation planning, safe handoff drafting, and schema-valid navigator JSON.

The first fine-tuning pilot was valuable because it proved the full loop: generate teacher data, train on Modal, merge the adapter, convert to GGUF, serve locally, and run the eval harness without cloud inference. But the result was not a clean win. The pilot made the model better at shape and field retention, but it regressed competence to `11/50` on the locked 50-case eval.

That failure was useful. It showed that training loss and JSON validity were not enough. The dataset had taught format more than judgment. It had too few examples for some failure modes. Some rows were not aligned tightly enough to the real harness. And the eval was punishing behaviors that looked safe in prose but violated the exact scorer or product contract.

The v2 dataset was a better answer. It used a stronger teacher model to generate synthetic, validated rows aligned to the actual Figment prompt and repair tasks. It kept locked eval cases out of training. It added more repair rows and failure-class coverage. The v2 local model improved to `33/50` on the locked 50-case eval, with `50/50` final validation.

Then v3 changed the question again.

Rather than only optimizing the locked 50-case eval, I created a 150-case field-workflow holdout. That holdout asked whether Figment helped the real workflow: rural clinic intake, disaster triage, ASR-like confirmed text, low-resource constraints, radio handoff, SBAR usefulness, and source-card discipline.

On that holdout, v3 reached `107/150` competence, with `93/150` raw local-model successes, `14/150` focused repair successes, `2/150` full fallbacks, and `148/150` final validation.

That sounded good, and in many ways it was. But the failure distribution mattered more than the headline score. The handoff layer was weak: radio handoff and SBAR usefulness were exactly where the model still needed help.

That is where the next two days of work changed the project.

## Sometimes The Benchmark Is Wrong

The most valuable bug I found was not in the model. It was in the scoring and deterministic rule path.

The old holdout treated some negated phrases too bluntly. A sentence like "no chest pain reported" could still trigger a chest-pain-related signal because the matcher saw the words and missed the negation. That is the kind of bug that can make a benchmark look tougher while actually making it less faithful.

I did not mutate the original frozen holdout. Instead, I created a corrected scoring view with a manifest. It changed exactly six cases and kept the original and corrected hashes visible. That mattered because the point of an eval is trust. If the target moves, readers should be able to see how and why.

This became a new rule for the project: do not train your way around a bad benchmark. Fix the benchmark, leave a receipt, and rerun the model.

## Sometimes Prompting Harder Makes Things Worse

I also tried the obvious prompt fixes.

Some failures involved missing required observation ownership. So I tested stricter prompt contracts that made required observation IDs more explicit and more mandatory. In theory, that should have helped. In practice, one mandatory-observation prompt probe made the run worse.

That was a useful embarrassment.

It reminded me that a prompt is not a magic policy layer. If the model is already near the edge of a narrow behavior, adding more contract language can crowd the task, make it overfit the wrong cue, or shift attention away from the actual field workflow. Some failures needed better data. Some needed a clearer scaffold. Some needed the app to stop asking the model for things the app already knew. "Prompt harder" was not a general solution.

## V5 Through V14p Became A Curriculum Loop

The later local models were less like one big training run and more like an eval-driven curriculum.

V5 proved that the harness could keep the app safe even when the model was not carrying the work. It reached `150/150` final validation, but only `2/150` model competence. That result forced the right question: what would it take for the configured model, not the scaffolding, to own the fields?

V6 was the first big answer. Instead of regenerating everything from scratch, I built a corpus with targeted deltas plus replay rows from earlier versions. It reached `142/150` competence, `150/150` final validation, zero fallback, and far fewer deterministic patches.

V7 improved again: `148/150` competence, zero fallback, and only a handful of deterministic patches. On the corrected scoring view, it still had real misses, especially around postpartum-fever observation ownership and field-specific required-observation behavior. But now the failures were small enough to inspect case by case.

The v8-v14p loop kept narrowing those misses. The data was not generic "more medical examples." It was targeted rows for multi-rule observation ownership, postpartum-fever required observations, source/support cards, visible-field closure, and focused repair behavior. Some versions improved the exact metric. Some did not. A few looked nearly identical. That was frustrating, but it was also evidence that the eval had become specific enough to resist hand-wavy progress stories.

The strongest current run is v14p repair-union: `150/150` competence, `150/150` expected labels, and `150/150` final validation on the corrected 150-case holdout, with zero deterministic patches and zero fallback. The nuance matters: raw configured-model success is still `146/150`; four cases are resolved by focused model repair, and eight fields are marked as model-repaired rather than model-raw.

That is a much better result than the early local runs, but it is not the same claim as "the raw model passed everything." It proves something narrower and more useful:

The local 4B Figment system can complete this corrected field-workflow eval with model-owned output and model repair, without deterministic patching or full fallback, while preserving red flags, source discipline, and handoff constraints.

That is the kind of claim I can actually defend.

## The Product Surface Had To Catch Up

Another thing I learned: evidence is not enough if the product surface does not make the evidence legible.

Figment started as a fairly functional Gradio app. It had the pieces: intake, rules, retrieval, navigator output, trace. But it felt more like a harness than a field tool.

The later UI work moved it to a custom Gradio Server surface with a "Field Kit Workbench" feel. The important part was not just that it looked better. The important part was that the user workflow became clearer without changing the model harness contract. The named API endpoints, intake/risk/retrieval/navigator/trace data shape, demo-case loader, and eval harness stayed stable.

That taught me a product lesson I wish I had internalized earlier. You can make a prototype more delightful without hiding the machinery that makes it trustworthy. The right UI did not bury the trace; it made the workflow easier to understand so the trace could matter more.

## Public Receipts Matter

The hackathon also changed how I think about artifact publishing.

It is one thing to say "I trained a local model." It is another thing to publish model artifacts, dataset cards, configs, eval traces, and schema-stable dataset viewers that let someone inspect the path. By the end of the later loop, the Hugging Face repos had public artifacts and dataset configs for v5 through v14p, with the v8-v14p corpora published and verified.

That matters for a small-model project because the interesting claim is rarely just the final score. The claim is the path: which rows were added, which cases were excluded, which eval was frozen, which scoring view was corrected, which artifacts were served, and which fallback paths were counted separately.

The competitor scan made that even clearer. Some Build Small projects had very legible demos. Dental SOAP, for example, is a strong direct comparison: guided intake, small Qwen model roles, deterministic safety sentinel, and printable handoff. ScrubData is technically strong in another direction. Figment's edge is not that it is the simplest demo. Its edge is the depth of the evidence trail: model versions, datasets, traces, failure accounting, and an app surface that shows how the answer was made.

That is also the risk. A deep evidence trail only helps if judges and users can understand it quickly. The demo still has to make the Backyard story obvious: here is the messy field intake, here are the red flags, here is what the small local model contributed, here is what the app refused to let it decide, and here is the handoff you can use.

## What I Would Tell Another Build Small Team

If I were giving advice to someone building with a small model in a high-stakes-ish workflow, I would say:

Start with the boundary, not the model. Decide what the model is allowed to own, what the app owns deterministically, and what a human must confirm.

Measure app safety and model competence separately. If the final app output passes because a scaffold saved it, count that as scaffold success, not model success.

Make provenance a product feature. Users and judges should be able to see which fields came from the model, repair, rules, retrieval, or fallback.

Let failures become curriculum, but only after checking whether the eval is fair. Some misses deserve training rows. Some deserve harness fixes. Some deserve a corrected benchmark.

Do not assume a stricter prompt is a better contract. Verify it with the same eval, and be willing to throw it away.

Publish the receipts. Scores are more credible when the artifacts, datasets, traces, and manifests exist outside your laptop.

## The Lesson I Am Taking From Build Small

Before this project, I would have described small-model product work mostly in terms of parameter count, latency, hardware, and model quality. Those still matter. But Figment made me think about "small" differently.

Small is also a design discipline.

It means narrowing the model's job until it can be checked. It means using deterministic rules where determinism is safer. It means making retrieval explicit. It means refusing to count fallback as model competence. It means keeping traces detailed enough that a judge, user, or future builder can see what happened. It means letting the model contribute where language and prioritization matter, while keeping safety-critical floors outside the model's control.

The first version of this post ended with "make the next eval sharper." After v5 through v14p, I would say it a little differently:

Make the next failure smaller, clearer, and harder to hide.

That is the thing Figment taught me. Useful small-model apps are not small because they ask less ambitious questions. They are small because they are honest about where the model belongs, and because that honesty gives you a way to improve.
