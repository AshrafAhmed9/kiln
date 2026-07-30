# ADR-012: The learning loop — Kiln is a curriculum, not just a build

**Status:** Accepted (Revision 1.1)

**Decision:** each phase follows study → derive by hand → implement → teach
back, and isn't done until the teach-back step is. The phase's ★ reading-list
item is read before the branch opens; mathematical components (RoPE, GQA,
softmax numerics, the rejection-sampling acceptance rule, quantization
scale/zero-point) are derived by hand in `docs/learning/phase-NN.md` before
being coded; phase N+1 doesn't open until phase N can be explained unaided
from that note's summary.

**Why:** the project's value to Ashraf is the learning as much as the
artifact, and code that transcribes an understanding you don't have yet
produces exactly the "it passes and I'm not sure why" failure mode ADR-011
forbids. See project memory `kiln-learning-loop`.
