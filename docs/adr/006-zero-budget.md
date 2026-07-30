# ADR-006: Zero-budget constraint

**Status:** Accepted (Revision 1.1)

**Decision:** Kiln must be completable without spending money. Part I runs on
the existing MacBook (CPU-only) with GitHub Actions free tier for CI. Part II
GPU work runs on Kaggle's free ~30 GPU-hrs/week T4 (and dual-T4 for Phase 12).
Nightly GPU perf-regression CI (plan v1.0 Phase 11) is **not achievable
free** — no free GPU CI runner exists — so it's replaced by manual, versioned
benchmark runs per phase, with seed and Kaggle instance type recorded in
`BENCHMARKS.md`. Phase 14 drops Vertex AI (not free); Part IV serves on an
always-free CPU-only VM tier rather than a paid GPU instance.

**Why:** the project runs during a job search and must not compete with it
for money. See project memory `kiln-zero-budget-constraint`.

**Consequence accepted knowingly:** no continuous nightly GPU regression
tracking, and Part IV's public product serves at CPU speed, not GPU speed.
Both are stated honestly wherever they'd otherwise be implied.
