<!--
Sync Impact Report
===================
Version change: template → 1.0.0

Bump rationale: Initial ratification. Adapted from the shared five-principle governance
framework and retargeted to the domain of this project: an Image Segmentation →
Metrology pipeline that detects animals in COCO images, localizes animal eyes, and
measures inter-eye distances. Principle framework (Code Quality, Testing, UX/Output
Consistency, Performance, Language & Documentation) is preserved; domain-specific
clauses, rationales, and performance targets are rewritten for this project.

Modified principles: all five retargeted to the segmentation/metrology domain.
Added sections: none (same structure: Additional Constraints, Workflow, Governance)
Removed sections: none
Follow-up TODOs: none

Note: This Sync Impact Report is temporary scratch for amendment review and may be
removed before committing the constitution file.
-->

# Image Segmentation Metrology Pipeline Constitution

## Core Principles

### I. Code Quality

Code quality is a non-negotiable baseline. All code merged into the main branch MUST pass
automated static checks (linter, formatter, type checker) with no errors; warnings MUST be
resolved before merge or explicitly documented with a justified waiver. Every module MUST have a
single, clear responsibility (e.g. COCO selection, segmentation inference, eye localization,
metrology, evaluation, visualization), and public interfaces MUST be documented with their
purpose, inputs, outputs, coordinate conventions (pixel origin, axis direction, units), and
exceptional behavior. Commented-out dead code, leftover debug output, and unused dependencies MUST
NOT be committed. All dependencies MUST be declared with pinned versions, and any new dependency
(model weights, inference runtime, CV library) MUST be evaluated for maintenance status, license,
and security before adoption.

**Rationale**: The pipeline produces quantitative measurements consumed as results; low-quality or
undocumented code silently corrupts coordinates, distances, and evaluation numbers that are hard
to debug after the fact.

### II. Testing Standards

Testing is a delivery requirement, not an option. Core measurement logic (Euclidean distance
between eyes, inter-animal right-eye distance, pixel-to-real-unit scaling) MUST have unit tests
covering known-coordinate cases, boundary conditions (coincident points, zero distance, missing
eye), and numerical correctness. Pipeline-level behavior — COCO filtering (images with ≥2
animals), segmentation output shape/format, eye-localization fallbacks, and CSV output schema —
MUST have integration tests on a small fixed sample set. Any bug-fix commit MUST first add a test
that reproduces the bug. The test suite on the main branch MUST pass in full before merge, and
line coverage for the metrology and evaluation modules MUST be ≥ 80%.

**Rationale**: Measurement errors are easy to introduce and hard to notice; only repeatable tests
guarantee that geometric and evaluation formulas stay correct across changes.

### III. Output & Visualization Consistency

User-facing outputs (annotated overlays, measurement CSV, evaluation reports) MUST maintain
consistent terminology, schema, and presentation. Every annotated image MUST render masks, eye
points, measurement lines, and numeric values using a consistent color/label scheme, with left eye
and right eye unambiguously distinguished. The measurement CSV MUST carry a uniform set of required
fields: image id, animal instance id, animal class, left-eye (x,y), right-eye (x,y), inter-eye
distance, and (where applicable) inter-animal right-eye distance and unit. Coordinates MUST use a
single declared convention and distances a single declared unit (pixels by default; real units
only with a stated scale). Error and empty states (no animals, only one animal, undetected eyes)
MUST produce clear, actionable messages, not bare stack traces.

**Rationale**: The deliverable must be visually verifiable and machine-parseable; inconsistent
labels or schema make results unreviewable and untrustworthy.

### IV. Performance & Robustness Requirements

The system MUST meet measurable, reproducible targets. Segmentation plus eye localization for a
single image MUST complete within a documented budget on the declared reference hardware
(target ≤ 5 s/image on CPU, ≤ 1 s/image on GPU, P95), and the budget MUST be recorded alongside
results. Runs MUST be deterministic where possible: random seeds fixed, model weights and
versions pinned, and the same input MUST yield the same measurements. The pipeline MUST degrade
gracefully on hard cases (no eyes found → fall back per the documented strategy and flag the
record) and MUST NOT abort the whole batch due to a single failing image. Every performance claim
MUST be backed by a recorded measurement, not an estimate.

**Rationale**: Metrology results are only credible if they are reproducible and the compute cost is
quantified; non-deterministic runs make evaluation and verification meaningless.

### V. Language & Documentation

This constitution document MUST be authored in English. All other artifacts — specifications
(spec), plans (plan), tasks (tasks), user-facing documentation (README, result reports), and
report copy — MUST be written in Traditional Chinese (zh-TW). Code identifiers and technical
keywords MAY remain in English, but their accompanying documentation MUST follow the applicable
language rule for its artifact type. Documentation MUST be kept in sync with the implementation; in
particular, the chosen AI model, the evaluation formulas (IoU, Dice, NME), and the measurement
formulas MUST be documented and match the code. A commit that changes interface or behavior without
updating the corresponding documentation MUST be treated as incomplete.

**Rationale**: The reviewing audience reads Traditional Chinese, so delivered artifacts use zh-TW;
the constitution is kept in English as a stable, portable governance reference.

## Additional Constraints & Compliance

- Data source: The system MUST use the COCO dataset (or an authorized subset); only images
  containing two or more animals are in scope, and the selection logic MUST be explicit and
  reproducible.
- Eye-localization honesty: COCO provides no animal-eye annotation. The strategy used (keypoint
  model, SAM2 prompt refinement, or geometric heuristic fallback) MUST be documented, and any
  heuristic/estimated eye position MUST be flagged as such in the output.
- Model transparency: The chosen segmentation model, its rationale, and its evaluation criteria
  (formulas) MUST be stated; measurements derived from low-confidence detections MUST be marked.
- Verification: Measurements MUST be verifiable by overlay inspection and, where ground truth is
  manually annotated, by reported pixel error / NME against that ground truth.
- Licensing & privacy: Model weights and datasets MUST be used within their licenses; no
  unnecessary personal data is collected (COCO animal imagery only).

## Development Workflow & Quality Gates

- Every change MUST go through code review, and the reviewer MUST confirm conformance with each
  principle of this constitution.
- Pre-merge gates: static checks pass, tests green, coverage target met, documentation (model
  choice, evaluation formulas, measurement formulas) updated in the language required for its
  artifact type.
- Any complexity that deviates from the principles MUST be explicitly justified in the PR
  description; unjustified complexity MUST be rejected.
- Accuracy-sensitive changes (segmentation model/weights, eye-localization strategy, distance or
  scaling formulas) MUST be flagged and MUST state what was verified and against what reference.

## Governance

This constitution supersedes other development conventions; where a convention conflicts with this
constitution, this constitution prevails.
Amendment procedure: Any amendment MUST be proposed in writing, stating the change, rationale, and
impact, and takes effect only after approval by the project owner.
Versioning policy (semantic versioning):
- MAJOR: Removal or redefinition of existing governance clauses/principles (backward incompatible).
- MINOR: Addition of a principle or section, or material expansion of existing guidance.
- PATCH: Wording clarifications, typo fixes, and other non-semantic refinements.

Compliance review: All PRs and reviews MUST verify conformance with this constitution; at least
quarterly, the principles MUST be reviewed for continued applicability. When an amendment takes
effect, the version number and amendment date MUST be updated and dependent templates and
documentation synced as needed.

**Version**: 1.0.0 | **Ratified**: 2026-10-22 | **Last Amended**: 2026-10-22
