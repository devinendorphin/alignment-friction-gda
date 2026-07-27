# alignment-friction-gda

**Register: formal / empirical.** A published data release with a DOI
([10.5281/zenodo.20016461](https://doi.org/10.5281/zenodo.20016461)). The released CSVs are
the record; the paper cites them.

**Thesis:** the Gradient Decomposition Assay — a CSV-backed measurement of alignment
friction under direct constraint and counterfactual reframing across five frontier model
families. This is the empirical reduction of the `crownfull` v2.1 architecture: the point
where a speculative design was cashed out into something measurable.

## Repo-specific discipline

- **Released data is immutable.** The canonical CSVs and JSONL back a published paper.
  Never edit them to fix an inconsistency — add an erratum.
- **Reproduction must keep passing.** `GDA_Reproduction_Notebook.ipynb` regenerates every
  table and verifies each value to two decimals. Any change that touches data or derivation
  gets checked against it.
- **Preserve the record, including its warts.** The two executed prompt typos (`eupisms`,
  `eupistic`) and the vector-6 fresh-context caveat are deliberately kept. They are part of
  the experimental record, not bugs to tidy.
- **Respect the Claim-Status Ledger** (Appendix F): documented provenance, empirical result,
  AI-generated framing, and future-work hypothesis are different things. Do not let a
  framing claim drift into sounding empirical.
- **Invalid runs stay categorized.** Parse errors are evaluator/transport-side; nonsensical
  outputs are substrate-side behavior. Keep them separate.

## The harness

The canonical working agreements, the atlas of all 20 repos, and the shared glossary live in
**`devinendorphin/claude-at-claude`**. Pull it in when you need the full map:

```
add_repo devinendorphin/claude-at-claude
```

This container is ephemeral, so anything that matters gets committed *this turn*. Be a
collaborator rather than a cheerleader, and run a disconfirming test on primed claims.
Endorphin works from a phone and often dictates while walking — expect speech-to-text
artifacts, and mark guessed corrections `[?original→guess]`.
