# Publication overlap analysis

This document tracks which experimental publications can be safely combined
in global fits, and which would lead to double-counting.

---

## B0 -> K*mumu: CMS publications

### Publications

| Key | arXiv | sqrt(s) | Lumi | Observables | q2 bins (GeV2) |
|---|---|---|---|---|---|
| Khachatryan:2015isa | 1507.08126 | 7+8 TeV | 20.5/fb | FL, AFB, dBR/dq2 | 1-2, 2-4.3, 4.3-8.68, 10.09-12.86, 14.18-16, 16-19 |
| CMS:2017ivg | (CMS PAS) | 8 TeV | 20.5/fb | P1, P5' | 1-2, 2-4.3, 4.3-6, 6-8.68, 10.09-12.86, 14.18-16, 16-19 |
| CMS:2017rzx | 1710.02846 | 8 TeV | 20.5/fb | P1, P5' | same 7 bins |
| CMS:2024atz | 2411.11820 | 13 TeV | 140/fb | FL, P1-P3, P4', P5', P6', P8' | 1.1-2, 2-4.3, 4.3-6, 6-8.68, 10.09-12.86, 14.18-16 |

### Relationships

**CMS:2017ivg vs CMS:2017rzx** -- SAME measurement (preliminary vs published).
Both are analysis BPH-15-008 on the same 8 TeV dataset. Same central values;
the published version (2017rzx) has updated systematics and includes per-bin
P1-P5' 2x2 correlation matrices. **Never use both. Use CMS:2017rzx only.**

**Khachatryan:2015isa vs CMS:2017rzx** -- same 8 TeV dataset, different observables.
2015isa measures FL, AFB, dBR/dq2. 2017rzx measures P1, P5'. No observable
overlap, so combining is safe (standard practice; no cross-correlations provided).

**8 TeV papers vs CMS:2024atz** -- independent datasets.
Run 1 (7/8 TeV) vs Run 2 (13 TeV). No shared events. Safe to combine even
for the same observables (FL, P1, P5').

### Observable-by-observable overlap

| Observable | 2015isa (8 TeV) | 2017rzx (8 TeV) | 2024atz (13 TeV) | Overlap? |
|---|---|---|---|---|
| FL       | yes | -   | yes | safe (independent datasets) |
| AFB      | yes | -   | -   | unique to 2015isa |
| dBR/dq2  | yes | -   | -   | unique to 2015isa |
| P1       | -   | yes | yes | safe (independent datasets) |
| P2       | -   | -   | yes | unique to 2024atz |
| P3       | -   | -   | yes | unique to 2024atz |
| P4'      | -   | -   | yes | unique to 2024atz |
| P5'      | -   | yes | yes | safe (independent datasets) |
| P6'      | -   | -   | yes | unique to 2024atz |
| P8'      | -   | -   | yes | unique to 2024atz |

### Curated selection (c9_vs_q2.py)

```
CMS:2024atz         # 13 TeV -- FL, P1-P3, P4'-P8'
CMS:2017rzx         # 8 TeV  -- P1, P5' (with correlations)
Khachatryan:2015isa # 8 TeV  -- FL, AFB, dBR/dq2
```

No double-counting. CMS:2017ivg excluded (superseded by CMS:2017rzx).

### Notes on flavio measurements.yml entries

- Khachatryan:2015isa appears in 4 blocks: two for the standalone 8 TeV
  4.3-6 bin (FL+AFB and BR separately), and two for the 7+8 TeV combination
  (FL+AFB and BR, using merged 4.3-8.68 bin).
- CMS:2017rzx appears in 7 blocks, one per q2 bin, each with P1+P5' and
  their 2x2 correlation matrix.
- CMS:2017ivg appears in 2 blocks: all bins of P1 in one, all bins of P5'
  in another, without correlations.
- CMS:2024atz appears in 6 blocks, one per q2 bin, each with all 8
  observables and the full 8x8 correlation matrix.
