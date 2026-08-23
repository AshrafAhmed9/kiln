# Phase 10 — derivation notes (speculative decoding)

## The problem this solves

Normally, generating each new word requires one full pass through the
whole (large, slow) model. Speculative decoding's idea: let a much
smaller, faster "draft" model guess several words ahead on its own, then
check all of those guesses against the real (large) model in a *single*
pass, instead of one pass per word. Checking several guesses at once is
almost as cheap as checking one, so if the draft model's guesses are often
right, this saves a lot of the big model's time -- while still, provably,
producing exactly the words the big model would have chosen anyway.

## Why the acceptance rule is the part that has to be exactly right

Just accepting the draft model's guesses whenever they seem plausible
would quietly change what the model actually says over time, in ways that
would be very hard to notice by eye. The rejection-sampling rule this
phase implements is specifically designed so that, whatever the draft
model guesses, the *final* sequence of words has exactly the same
probability of occurring as if the big model had generated every word
itself, one at a time, with no shortcuts. The rule: for a guessed word,
compare how likely the *big* model thought it was (call this `p`) against
how likely the *small* model thought it was when it guessed it (call this
`q`). Accept the guess with probability `min(1, p/q)` -- if the big model
agreed at least as strongly as the small model did, always accept; if the
big model was less confident, accept only some of the time, proportional
to how much less confident. The moment a guess is rejected, a replacement
word is drawn instead, from the *leftover* difference between what the big
model wanted and what the small model already accounted for (`max(0, p -
q)`, renormalized) -- not from the big model's plain distribution, and not
from the small model's. This specific leftover-distribution choice is
exactly what makes the whole scheme mathematically exact, and it's the
part that would be easy to get wrong by intuition alone (guessing "just
resample from the big model" seems reasonable, but it isn't -- it would
double-count the words the small model already got a fair chance at
proposing).

## Why the cleanest test is the greedy case, not the random-sampling case

When decoding always just picks the single best word (temperature 0, no
randomness at all), the general accept-or-replace rule above collapses to
something very simple and fully deterministic: accept the small model's
guess exactly when it happens to match what the big model would have
picked anyway; the instant it doesn't match, replace it with the big
model's actual pick, and stop guessing further ahead for this round. There
is no randomness left anywhere in this special case -- which means
speculative decoding's output, in greedy mode, must be *exactly*,
token-for-token identical to just running the big model alone, every
single time. That's a claim that can be checked directly and completely,
with no statistics involved, which is why it's the test this phase leans
on hardest. The fully general case (real randomness, real temperature) is
proven correct by the math in the original papers (Leviathan et al.; Chen
et al.), and this project's implementation follows that math directly --
but proving it holds in general, empirically, would need a much larger
statistical comparison (many thousands of samples, compared distribution
to distribution) that a real GPU run is better suited to than this
CPU-only, offline session.

## An honest scope decision: no persistent KV cache in this implementation yet

Real speculative decoding keeps using the KV cache the same way normal
generation does. Doing that here would require being able to roll a
cache *back* when a guessed word gets rejected partway through a round --
and Phase 3's cache was deliberately built append-only (grow, never
shrink), since that was the right amount of complexity for what Phase 3
needed. Rather than quietly bolt a rollback feature onto that cache
without the same testing rigor the rest of this project holds itself to,
this phase recomputes the whole context from scratch each round instead.
That's slower (real speculative decoding's whole point is being fast), but
it keeps the part actually being taught in this phase -- the acceptance
rule -- correct and clearly isolated from a cache feature that doesn't
exist yet. Making it fast again is named here as real, deferred work, not
hidden.
