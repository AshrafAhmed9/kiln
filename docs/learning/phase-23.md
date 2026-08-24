# Phase 23 — one shared compute step, many private histories

Continuous batching does not mean one conversation can read another one's
cache. It means several requests can put one new token each into a single
matrix-multiplication batch because they all use the same weights. Their
attention state still has to stay separate.

`ForwardDecodeBatch` follows that split directly. It computes embeddings,
normalization, Q/K/V projections, output projection, and MLP rows together.
For each row, it then appends the new key/value pair to that request's own
`KVCache` and runs attention against only that cache. The result is the same
math as calling cached decode once per request, with a batched outer shape.

The important proof is not that two HTTP requests return: both could return
from a secretly serial implementation. The C++ test starts two caches at
different lengths, runs one batched decode token, then repeats the same work
independently. Every output logit matches within the existing FP32 tolerance.

The API worker is deliberately narrow. It queues normal completions and lets
the scheduler decide when they run. Fresh prompts prefill separately; only the
later one-token decode steps batch. Streaming has not been redirected yet,
because sending each token to a client needs a tested queue/cancellation path
rather than treating an HTTP generator as a normal blocking response.
