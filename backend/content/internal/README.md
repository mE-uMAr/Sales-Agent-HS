# Internal content — never indexed

Nothing in this directory reaches the chatbot.

`app/knowledge/ingest.py` reads **only** `content/public`. This tree is not
scanned, not embedded, and not stored in Chroma, so there is no retrieval path
to it — a prompt-injection attempt cannot surface what was never indexed.

Put anything here that must not reach a visitor: cost structures, margins,
salary bands, partner agreements, unannounced work, pipeline notes.

Two rules keep this true:

1. **Never move a file from here into `content/public` without reading it.**
   The public tree is the trust boundary.
2. **Do not point `CONTENT_DIR` at this directory.** The ingester resolves every
   path and refuses anything outside `content/public`, but the setting is not a
   place to experiment.

`tests/test_leak_prevention.py` asserts that phrases appearing only in this
directory never appear in a retrieval result or a bot reply. If you add a file
here, consider adding a canary phrase from it to that test.
