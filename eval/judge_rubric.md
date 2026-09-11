# LLM-as-Judge Rubric — AmazonHelp Reply Quality

## Purpose

This rubric defines how the LLM judge evaluates draft replies produced by the
AmazonHelp pipeline. Each dimension is scored 1-5 independently, then a
pass/fail gate is applied.

## Judge Model

**Model**: `grok-3-mini` (different tier from the drafting model `grok-3`).
This tier difference is a deliberate mitigation for same-family self-preference
bias. **Limitation**: residual same-vendor bias is not fully eliminated by a
tier change alone — noted explicitly in REPORT.md "What's Misleading."

## Evaluation Dimensions

### 1. Groundedness (1-5)
*Is the reply grounded in real Amazon/AmazonHelp policy precedents, as seen in
the retrieved historical examples? Does it avoid inventing specifics
(refund amounts, timelines, policies) not supported by evidence?*

| Score | Criteria |
|-------|----------|
| 5 | Reply draws directly on retrieved precedents; makes no unsupported claims |
| 4 | Mostly grounded; one minor claim slightly beyond retrieved evidence |
| 3 | Mix of grounded and ungrounded statements |
| 2 | Largely ungrounded; multiple invented specifics |
| 1 | No connection to retrieved evidence; fabricates policies/amounts |

### 2. Relevance (1-5)
*Does the reply directly address the customer's actual ask? Does it avoid
answering a different question than the one posed?*

| Score | Criteria |
|-------|----------|
| 5 | Directly and completely addresses the customer's question/issue |
| 4 | Addresses the main ask; one minor tangent or omission |
| 3 | Partially relevant; addresses some but not the core issue |
| 2 | Mostly irrelevant; addresses wrong intent or is too generic |
| 1 | Entirely off-topic or completely ignores the customer's ask |

### 3. Factual Consistency (1-5)
*Is the reply internally consistent and consistent with what the retrieved
historical examples show? No contradictions or impossible claims.*

| Score | Criteria |
|-------|----------|
| 5 | Fully factually consistent; no contradictions |
| 4 | One minor factual awkwardness (e.g., imprecise phrasing) |
| 3 | One noticeable inconsistency or dubious claim |
| 2 | Multiple inconsistencies or one clearly wrong factual claim |
| 1 | Contradicts itself or makes false claims about Amazon policy |

### 4. Tone (1-5)
*Does the reply match AmazonHelp's observed brand voice: warm, professional,
empathetic, concise, action-oriented? Not defensive, not robotic.*

| Score | Criteria |
|-------|----------|
| 5 | Perfect AmazonHelp tone; empathetic, professional, actionable |
| 4 | Good tone with minor awkwardness |
| 3 | Acceptable but noticeably stiff, overly formal, or slightly cold |
| 2 | Poor tone: defensive, dismissive, or inappropriately casual |
| 1 | Wrong tone: aggressive, rude, or completely off-brand |

### 5. Actionability (1-5)
*Does the reply give the customer a clear next step? Does it specify what to
do, where to go, or what information to provide?*

| Score | Criteria |
|-------|----------|
| 5 | Clear, specific next step that the customer can immediately act on |
| 4 | Good next step; slightly vague on one detail |
| 3 | Next step exists but is generic or easy to miss |
| 2 | Vague or no clear next step |
| 1 | No actionable guidance whatsoever |

## Pass/Fail Gate

A reply **fails** (regardless of individual scores) if ANY of the following are true:
- It promises a specific refund amount, timeline, or policy detail **not found**
  in any retrieved historical example.
- It asks the customer to share a password, full credit card number, or SSN.
- It responds to the **wrong** customer issue entirely (relevance score = 1).
- It contains a factual claim that directly contradicts known Amazon policy
  (e.g., claims returns are accepted after 1 year).
- Score on **any single dimension** is 1.

## Aggregate Score

`aggregate = (groundedness + relevance + factual_consistency + tone + actionability) / 5`

Threshold for "acceptable reply": aggregate ≥ 3.5 AND no single dimension score of 1 AND pass/fail = PASS.

## Judge Prompt Template

See `eval/run_eval.py → _judge_prompt()` for the exact prompt sent to the judge model.

## Human-Judge Agreement Protocol

1. Evaluator independently scores the same 40-50 replies **before** seeing
   the judge's scores (no order bias).
2. Compute per-dimension quadratic-weighted Cohen's kappa between human and judge.
3. Report the aggregate kappa and flag dimensions with low agreement.
4. A kappa < 0.40 on any dimension is reported as a reliability concern.
