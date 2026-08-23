# Phase 18 — derivation notes (launch, users, and operations)

## What this phase asks for, and why it can't be simulated

Every earlier phase in this project has an honest CPU-only or
offline-environment substitute: no GPU, so kernels are written to spec and
deferred; no real training data, so a synthetic eval task suite stands in.
Phase 18 is different in kind, not just in degree. It asks for **real
users, a real public deployment, real incidents, and real retention
data**. There is no honest substitute for any of these. A fabricated user
count or an invented postmortem wouldn't be an honest simplification the
way "untrained toy model" or "CPU-simulated tensor parallelism" are — it
would be a fabricated claim about the world, indistinguishable from a lie
on a resume. The entire discipline this project holds itself to (state
exactly what's proven, exactly how, and exactly where the proof stops) is
worth nothing the moment a single number in this section is made up.

## What is actually true as of this session

- The engine has been deployed and verified inside a real Docker
  container, on this machine, built from `deploy/Dockerfile` — this
  actually caught two real bugs in the process (see docs/correctness.md):
  a static library missing position-independent code, which only breaks
  on Linux and not macOS, and a missing copy of installed console scripts
  in the final image stage. Both are fixed and re-verified.
- `deploy/docker-compose.yml` wires the engine to Prometheus (scraping
  the real `/metrics` endpoint added in this phase) and Grafana.
- **Nothing has been deployed publicly.** There is no live URL. There are
  no users, real or otherwise. There is no uptime history, no incident,
  no postmortem, and no retention number, because none of those things
  have happened.

## What would actually need to happen for this phase to be true

Deploying to a real cloud provider, pointing a domain at it, announcing
it somewhere real people would see it (Show HN, r/LocalLLaMA, etc.),
watching what real strangers do with it, and writing up what actually
happens — including anything that breaks. All of that is genuine future
work, gated on a decision Ashraf makes outside of this session (whether
and when to actually launch), not something any amount of further
offline work in this environment can substitute for.

## The one honest deliverable this phase can produce right now

A postmortem *template* — the structure a real incident writeup would
follow, ready to use the moment there's a real incident to write about.
See `docs/postmortems/TEMPLATE.md`. It contains no incident, because
there has been none.
