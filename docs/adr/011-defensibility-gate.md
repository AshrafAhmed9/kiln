# ADR-011: Defensibility gate

**Status:** Accepted (Revision 1.1)

**Decision:** Each phase's Definition of Done gains a page in
`docs/defense.md` — an explanation of the component, why it works, and what
it cost, written from memory before re-reading the code. No code enters the
repo that can't be derived on request. A phase isn't done until its defense
page exists.

**Why:** parity-green code that isn't understood is worse than a failing
test — it hides the gap until an interviewer finds it. See project memory
`kiln-minimal-defensible-code` and `kiln-learning-loop`.
