"""Detection-quality evaluation harness.

Measures how ACCURATE the Stage-2 VLM is against a labelled clip set — precision,
recall, F1 and a confusion matrix — so model / prompt / threshold changes are a
data-driven regression gate instead of intuition. See README.md in this package.
"""
