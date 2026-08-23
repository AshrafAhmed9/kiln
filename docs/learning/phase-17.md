# Phase 17 — derivation notes (the playground)

## Why a static page instead of the Next.js app the plan describes

Node and a real React toolchain were actually available while building
this, so this wasn't a case of "couldn't." It was a judgment call: a
single self-contained HTML file with plain JavaScript achieves the exact
same real, working feature (talk to the API, compare settings
side-by-side, watch text stream in) with no build step, no dependency
tree to keep working, and every line of it readable top to bottom in one
sitting -- which is the same "as simple as possible, easy to study"
standard this project holds its C++ to. A Next.js rewrite of this exact
page is a reasonable, real next step if this project's playground ever
needs more than what a static page can do (persistent accounts, a design
system shared with a larger site) -- but reaching for that complexity
before it's actually needed would be the same mistake the constitution's
non-goals list already warns against elsewhere in this project.

## Why the live comparison uses temperature, not quantization level

The plan's original vision for this phase is a side-by-side comparison of
different *quantization levels* and *speculative decoding on/off* --
genuinely the most interesting version of this feature, because it makes
Part II's accuracy-vs-speed tradeoffs visible and visceral. But that
comparison can only be real if the API actually has more than one servable
configuration to switch between, and building that (multiple loaded model
variants, or dynamic dequantization behind the API) is real backend work
that hasn't been done yet -- Part II's quantizer and speculative decoder
exist and are tested as standalone modules, not yet as alternate serving
paths the API can switch between per request. Rather than fake that
comparison with a toggle that doesn't actually change what's running
underneath, the playground compares something that genuinely is wired
end to end right now: two different temperature settings, run for real,
side by side, with real measured latency for each. The quantization/
spec-decode comparison is named here as the next real step, not
quietly skipped.
