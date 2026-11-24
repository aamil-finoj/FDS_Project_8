# Presentation split

Group of 6: Rohan, Ashwath, Joel, Aamil, Arnav, Advay. People were assigned at random.

The code and results are done, so the work left is **understanding, optimising and presenting**. The
project is split into six presentation segments. Each one is self-contained (it can be learned and
presented without anyone else's segment) and carries about the same amount of material to learn.

**How "equal" was measured.** Learning load is approximated by the words of write-up in the notebook that
a person has to understand (their section, its conclusion paragraph, and any subsection moved to them).
It is a proxy: it counts material, not difficulty. All six come out within about 6% of each other.

## 1. Segments

| Person | Segment | Covers (notebook) | Learning load |
|---|---|---|---|
| **Ashwath** | The problem and the exact answer key | §2 generative model, §3 (Part 1) and its conclusion | about 850 words |
| **Joel** | Why squared error fails | §4 (Part 2), §7.4 "why MSE cannot assess recovery", their conclusions | about 775 words |
| **Arnav** | The MDN and its design choices | §5 (Part 3), §7.3 variance floor, their conclusions | about 785 words |
| **Aamil** | MDN vs single Gaussian, head to head | §6 (Part 4) without the branch-probability cell, its conclusion | about 820 words |
| **Rohan** | Stress test: rare branches | §7.1 asymmetric prior, §7.2 initialisation, §7.5 sample-size & extreme imbalance stress test, the branch-probability cell from §6, Part 5 conclusion | about 850 words |
| **Advay** | Stress test: noise | §8 (Extension) and its conclusion | about 860 words |

The closing material (§10 **Limitations** and **Bottom line**, about 210 words) belongs to nobody in particular:
**everyone learns it** (see section 3).

## 2. Presentation order, inputs and handoffs

Each person only needs the one-line "given" from the person before them; nothing else.

| # | Person | Takes as given | Hands off with |
|---|---|---|---|
| 1 | Ashwath | Nothing | "We now know the exact answer for every x: two symmetric branches at ±√x." |
| 2 | Joel | The exact two-branch posterior | "Squared error returns the average, 0, which is never a right answer. We need a model that outputs a distribution." |
| 3 | Arnav | Why point estimates fail | "Here is that model: a mixture of Gaussians, trained by log-likelihood. Now let's test it." |
| 4 | Aamil | The MDN definition (M Gaussians) and the exact densities | "On normal data the MDN matches the exact answer and the single Gaussian can't. Now we break the setup." |
| 5 | Rohan | The MDN wins on balanced data; excess NLL is the score | "With 97:3 odds the rare branch is still found. Now what about noise?" |
| 6 | Advay | Everything above | The bottom line, then questions |

**What each person learns and presents:**
* **Ashwath:** Bayes' rule for p(t \| x); why Z(x) has no closed form and how the quadrature is stabilised; the four input regions and where the modes are (t = 0 and ±√x); the checks (integrates to 1, symmetric, tolerance-independent).
* **Joel:** the identity E[(f − t)²] = E[(f − E[t\|x])²] + E[Var(t\|x)], so the optimum is the conditional mean; why that is 0 for a symmetric prior and 0.8√x for q = 0.9; why equal MSE for every model (about 1.339 vs 0.473 at q = 0.9) hides large differences in NLL.
* **Arnav:** the mixture formula, softmax weights, softplus variances plus floor v_min; training setup; logsumexp vs the naive sum (−∞ vs −3105 at t = 30); why a floor wider than the true branch (v_min = 0.1) breaks recovery.
* **Aamil:** excess NLL as a KL estimate and why its *paired* standard error is the right uncertainty; choosing M on validation NLL (any M ≥ 2 ties, M = 3 kept); the head-to-head table (excess NLL 1.68 vs 0.03, W1 0.45 vs 0.04, 5-seed means); the densities at chosen x and where the MDN's error really sits (beyond the support, not the merged dip); calibration via the PIT (KS 0.24 vs 0.013); the forward-residual test (94% vs 35% within ±2σ, target 95%) and why the target is σ, not 0.
* **Rohan:** the asymmetric prior and why P(t > 0 \| x) = q exactly; branch-probability error and capture/recovery (mass, not component counts); results (60/60 recovered for M ≥ 2, M = 1 never); initialisation hardly matters at N = 10k; the §7.5 stress test (breaks at about 10 minority points, one dataset per N so the N = 2500 vs 10000 difference is a draw effect).
* **Advay:** noise sweep σ = 0.05 to 0.5; why excess NLL is worst at the smallest noise (branches get very sharp); why the single Gaussian looks "better" at high noise; the prior's jump at 0 and mixture limits.

## 3. Everyone must know

Everyone needs to be able to explain these, including the parts presented by others.

**The story in three sentences.** A secret number t is squared and noised: x = t² + ε. Squaring loses the sign,
so one x has two valid answers, +√x and −√x. A network trained with squared error returns their average (0,
which is wrong), while a network that outputs a *distribution* (the MDN) finds both.

**Shared vocabulary.**
* **Posterior** p(t \| x): all answers with probabilities. Computable exactly here, so it is the answer key.
* **MDN:** a mixture of M Gaussians whose weights, means and variances depend on x; M = 1 is the "single Gaussian".
* **Excess NLL** = model NLL − exact-posterior NLL (0 is perfect). **Capture/recovery** = does the model put mass on each branch.
* **q** = prior weight on t > 0 (0.5 means balanced). **v_min** = variance floor. **σ** = noise level.

**Shared setup:** a = 2, σ = 0.15, 10 000 training / 2 000 validation / 2 000 test points, a 3 × 64 tanh network, 5 seeds.

**Headline numbers** (README §5): same test MSE everywhere (about 1.339); excess NLL 1.68 (single Gaussian) vs 0.03 (MDN);
rare branch recovered in 60 of 60 runs for M ≥ 2 at up to 97:3; M = 1 never; both branches recovered at every noise level.

**Limitations and bottom line** (README §6, notebook §10): one dataset size and one network size were tested, so
much less data or more extreme imbalance may miss the rare branch (untested); "recovered" is our own lenient
threshold (capture ≥ 0.5); GPU training isn't bit-for-bit repeatable; the point is not "MDN beats MSE" but that MSE
answers a different question.

**Cross-check (do this before presenting).** Each person writes two questions an examiner could ask about the *next*
person's segment (Ashwath → Joel → Arnav → Aamil → Rohan → Advay → Ashwath) and asks them in a practice run.
If someone can't answer, that segment's owner revises their slides.

**Running things.** You rarely need to retrain: all figures are in `figures/` and all numbers in
`artifacts/results/*.csv`. A full retrain takes about 1.5 h; setup is in README §8.

## 4. Optional optimisation ideas (one per person)

Each is taken from the project's own stated limitations or next steps, and is only worth doing if time allows.
Mention results in your segment only if they are finished and checked.

| Person | Idea |
|---|---|
| Ashwath | Make the quadrature for Z(x) faster or vectorised without losing accuracy (check against the 400 001-point grid) |
| Joel | Add an absolute-loss (median) network next to the MSE network, to show it also lands off the branches. **Done: notebook §4.1** (result: off the branches for q = 0.5, on the majority branch for q = 0.9, never both) |
| Arnav | Try truncated mixtures to remove the about 0.7% of mass that leaks outside [−a, a]. **Done: notebook §5.5** (`truncate=True` cuts leaked mass to `0.000` everywhere and reduces baseline excess NLL from 0.030 to 0.018 nats, better on every seed; W1 unchanged) |
| Aamil | Choose M by held-out NLL across M ∈ {1, 2, 3, 5} and justify the pick. **Done: notebook §6.1** (chosen on validation NLL; any M ≥ 2 ties, M = 3 kept), plus §6.3 region breakdown and §6.4 PIT calibration |
| Rohan | Test smaller training sets and more extreme imbalance. **Done: notebook §7.5** (N ∈ {1000, 2500, 10000} × q ∈ {0.9, 0.97, 0.99}: recovered down to about 30 minority points; breaks at about 10 (N = 1000, q = 0.99, 2 of 5 seeds lose it); spread init does not rescue that case but helps at N = 10000, q = 0.99) |
| Advay | Look at the prior's jump at t = 0 at high noise, which a smooth mixture can only approximate. **Done: notebook §8.3** (the step exists at every noise level; the MDN draws it as a ramp about 0.1–0.2 wide, costing at most about 0.01 nats (0.002–0.012) with branch probabilities unaffected; M = 8 does not sharpen it) |
