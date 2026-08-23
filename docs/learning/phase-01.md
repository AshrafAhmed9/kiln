# Phase 1 — derivation notes (written before implementation)

## Safetensors format

A `.safetensors` file is: `[8 bytes: header length N, little-endian u64]
[N bytes: JSON header] [rest: raw tensor bytes]`.

The JSON header is a flat object: one key per tensor name, mapping to
`{"dtype": "...", "shape": [...], "data_offsets": [start, end]}`, plus an
optional `__metadata__` key we ignore. `start`/`end` are byte offsets
*relative to the end of the header* (i.e., relative to the start of the raw
data region), not absolute file offsets.

Why this format is a good mmap fit: it's just a length-prefixed JSON
directory followed by a flat byte blob — no compression, no per-tensor
framing. That means `mmap`-ing the whole file and handing back
`base_ptr + header_size + data_offsets[0]` for each tensor is a genuine
zero-copy read: no allocation, no parsing of the tensor data itself, just
arithmetic on a pointer. The only real parsing work is the one JSON header.

Corrupted-header handling: if the declared header length runs past the file
size, or the JSON doesn't parse, or a tensor's `data_offsets` runs past the
data region, reject at load time rather than at first read — a bad file
should fail loudly at `Load()`, not silently hand back a dangling view.

## Byte-level BPE (the tokenizer)

Goal: turn a string into a sequence of integer token IDs, and back.

**The vocabulary is byte-level**, not character-level: every one of the 256
possible byte values is remapped through a fixed byte↔unicode-character
table (GPT-2's trick) so that *any* input string — including raw bytes that
aren't valid UTF-8 on their own — always has a representation. This is why
byte-level BPE never has an "unknown token" problem the way word-level
tokenizers do: everything is representable as some sequence of these 256
base symbols.

**Training already happened; we only need inference (merge application):**
`tokenizer.json` ships two things — a `vocab` (string → id) and an ordered
list of `merges` (pairs of strings that get merged into one, in the order
they were learned). Encoding one word is:

1. Start with the word as a sequence of the byte-level base symbols.
2. Look at every *adjacent pair* in the current sequence. For each pair that
   appears in the `merges` list, note its rank (earlier in the list = higher
   priority, i.e., it was a more frequent pair during training).
3. Merge the *single* pair with the best (lowest) rank into one symbol.
4. Repeat from step 2 with the new, shorter sequence, until no adjacent pair
   in the sequence appears in `merges` at all.
5. Map each final symbol to its vocab ID.

The reason it's "lowest rank wins" and not "leftmost pair" or "longest
pair": the merge list's order **is** the priority — it's the order those
merges were discovered during BPE training, which tracked pair frequency.
Applying merges in that fixed order at inference time is what makes
tokenization deterministic and matches what the model was actually trained
on. Getting the *tie-breaking rule* wrong (e.g., merging in sequence order
instead of rank order) produces a tokenizer that runs and produces plausible
looking tokens but silently disagrees with the reference on real text — the
exact "parity-green but wrong" trap the constitution is designed to catch,
except here it'd be upstream of the parity harness entirely, so this has to
be gotten right by construction, not caught by a diff.

**Word splitting before BPE:** real tokenizers first split text into
"pre-token" chunks (roughly: words and punctuation, via a regex) and run BPE
independently within each chunk — merges never cross a pre-token boundary.
Phase 1 implements this with the standard GPT-2-style splitting pattern.

**Decoding** is simpler: map each ID back to its vocab string, concatenate,
then invert the byte-level remapping back to raw bytes.

**What I'd get wrong without this derivation:** I would have guessed
"merge whichever pair is leftmost" (wrong — it's rank/priority order) or
tried to skip the byte-level remapping table (wrong — that's precisely what
makes the tokenizer total over all possible byte strings).
