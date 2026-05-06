# Parameter Derivation from Reference Implementation

All HEonGPU example files in `example/ranking_and_sorting/` implement algorithms from:

> **"Efficient Ranking, Order Statistics, and Sorting under CKKS"**
> Mazzone, Everts, Hahn, Peter — USENIX Security 2025

Parameters are derived from the paper's OpenFHE reference implementation at
`openfhe-statistics/src/`. This document traces **every parameter choice** in
each example file back to the exact line in the reference code.

---

## Table of Contents

1. [Reference Implementation Overview](#1-reference-implementation-overview)
2. [Shared Constants](#2-shared-constants)
3. [19\_ckks\_minimum — Homomorphic Minimum](#3-19_ckks_minimum)
4. [20\_ckks\_median — Homomorphic Median](#4-20_ckks_median)
5. [21\_ckks\_ranking\_multi — Multi-Ciphertext Ranking](#5-21_ckks_ranking_multi)
6. [22\_ckks\_sorting\_paper — Homomorphic Sorting](#6-22_ckks_sorting_paper)
7. [23\_ckks\_ranking\_tie\_correction — Single-CT Ranking (Paper-Spec)](#7-23_ckks_ranking_tie_correction)
8. [24\_ckks\_ranking\_tie\_correction\_extended — Single-CT Ranking (Extended)](#8-24_ckks_ranking_tie_correction_extended)
9. [Key Differences: OpenFHE vs HEonGPU](#9-key-differences)

---

## 1. Reference Implementation Overview

### Source files

| Reference file | Purpose |
|---|---|
| `test-ranking.cpp` | Ranking: single-CT, multi-CT (Chebyshev), multi-CT (f,g) |
| `test-sorting.cpp` | Sorting: single-CT and multi-CT, always with tie correction |
| `test-minimum.cpp` | Minimum: single-CT (Chebyshev) and multi-CT (f,g) |
| `test-median.cpp` | Median: single-CT (f,g) and multi-CT (f,g) |
| `utils-basics.cpp` | `generateCryptoContext()` — CKKS context creation |
| `utils-eval.cpp` | `depth2degree()`, `compare()`, `signAdv()`, `indicator()`, `indicatorAdv()` |

### Context generation (`utils-basics.cpp`)

All reference test functions share a single context generator:

```
ScalingTechnique  = FLEXIBLEAUTO
KeySwitchTechnique = HYBRID
SecretKeyDist     = UNIFORM_TERNARY
SecurityLevel     = HEStd_128_classic (when ringDim=0, auto-selected)
ScalingModSize    = decimalPrecision
FirstModSize      = integralPrecision + decimalPrecision
```

### `depth2degree` mapping (`utils-eval.cpp`)

This is the paper's bijective mapping between multiplicative depth consumed
by a Chebyshev evaluation and the resulting polynomial degree:

| depth | degree |
|------:|-------:|
| 3     | 2      |
| 4     | 5      |
| 5     | 13     |
| 6     | 27     |
| 7     | 59     |
| 8     | 119    |
| 9     | 247    |
| 10    | 495    |
| 11    | 1007   |
| 12    | 2031   |
| 13    | 4031   |
| 14    | 8127   |

These are NOT powers-of-2 minus 1. HEonGPU's BSGS Chebyshev evaluation
has a different depth-to-degree correspondence (roughly `ceil(log2(d+1))`
levels for degree `d`), but the degrees from this table are used as the
target approximation quality.

---

## 2. Shared Constants

### Comparison functions (`utils-eval.cpp`)

**Chebyshev sign** (`compare`): Approximates the half-comparison function
`f(x) = 1 if x>err, 0.5 if |x|<=err, 0 if x<-err` with default `err=0`.

**f,g composition** (`signAdv`): Five sequential degree-7 polynomial evaluations.
- g(x) = (4589x − 16577x³ + 25614x⁵ − 12860x⁷) / 1024
- f(x) = (35x − 35x³ + 21x⁵ − 5x⁷) / 16
- f_final(x) = f(x)/2 + 0.5 (shifts output from [-1,1] to [0,1])
- Applied as: g^dg → f^(df-1) → f_final
- Each degree-7 polynomial costs 4 levels in OpenFHE's Paterson-Stockmeyer

**Indicator** (Chebyshev): `1{a1 <= x <= b1}` over domain `[a, b]`

**Indicator (f,g)** (`indicatorAdv`): Two signAdv evaluations + 1 multiply

### Block sizes

| Algorithm | single-CT block | multi-CT block L |
|---|---|---|
| Ranking | N (vector length) | 128 |
| Sorting | N (vector length) | 256 |
| Minimum | N (vector length) | 128 (Cheby) / 256 (f,g) |
| Median  | N (vector length) | 256 |

---

## 3. 19\_ckks\_minimum

**Algorithm**: Order statistic (Algorithm 4) for k=1.
Computes `rank(v) → indicator(rank, target=1) → one-hot mask`.

**Reference**: `test-minimum.cpp`, functions `testMinimum` (single-CT) and
`testMinimumMultiCtxt` (multi-CT Chebyshev).

### Reference parameters (single-CT Chebyshev, `testMinimum`)

```
integralPrecision  = 1
decimalPrecision   = 59
multiplicativeDepth = compareDepth + indicatorDepth
numSlots           = N * N
```

| N range  | compareDepth | indicatorDepth | degree_C | degree_I | depth |
|---------:|:------------:|:--------------:|---------:|---------:|------:|
| N <= 32  | 7            | 7              | 59       | 59       | 14    |
| N <= 256 | 9            | 7              | 247      | 59       | 16    |

**Reference**: `test-minimum.cpp` main(), lines 281–302.

### Reference parameters (multi-CT f,g, `testMinimumMultiCtxtAdv`)

For N > 256, the reference switches to f,g with L=256:

```
integralPrecision  = 1
decimalPrecision   = 59
multiplicativeDepth = 4*(dg_c + df_c + dg_i + df_i) + 3 + 2
```

| N range    | dg_c | df_c | dg_i | df_i | depth |
|-----------:|:----:|:----:|:----:|:----:|------:|
| N <= 2048  | 3    | 2    | 4    | 2    | 49    |
| N <= 8192  | 3    | 2    | 5    | 2    | 53    |
| N <= 16384 | 3    | 2    | 6    | 2    | 57    |

**Reference**: `test-minimum.cpp` main(), lines 340–361.

### HEonGPU implementation (`19_ckks_minimum.cpp`)

Covers single-CT Chebyshev only (N <= 128). Uses `n=131072`.

| N range | degreeC | degreeI | HEonGPU depth                    | scale | dnum |
|--------:|--------:|--------:|:---------------------------------|------:|-----:|
| N <= 32 | 59      | 59      | 1(TransR) + 6 + 1(norm) + 6 = 14 | 59    | 1    |
| N <= 128| 247     | 59      | 1(TransR) + 8 + 1(norm) + 6 = 16 | 59    | 1    |

The Chebyshev degrees (59, 247) match `depth2degree(7)` and `depth2degree(9)`
from the reference. The `decimalPrecision=59` is matched exactly.

**Depth formula derivation**:
- `1` for TransR mask (multiply_plain + rescale)
- `compare_levels = ceil(log2(degreeC))` for Chebyshev comparison
- `1` for indicator normalization (rank domain → [-1,1])
- `indicator_levels = ceil(log2(degreeI))` for Chebyshev indicator

---

## 4. 20\_ckks\_median

**Algorithm**: Order statistic (Algorithm 4) for k=ceil(N/2).
Uses f,g composition for both comparison and indicator.

**Reference**: `test-median.cpp`, function `testMedianAdv` (single-CT f,g).

### Reference parameters (single-CT f,g, `testMedianAdv`)

```
integralPrecision  = 1
decimalPrecision   = 59
multiplicativeDepth = 4*(dg_c + df_c + dg_i + df_i) + 6
```

| N    | dg_c | df_c | dg_i              | df_i | depth |
|-----:|:----:|:----:|:-----------------:|:----:|------:|
| 8    | 3    | 2    | 1 (=log2(8)/2)   | 1    | 34    |
| 16   | 3    | 2    | 2 (=log2(16)/2)  | 1    | 38    |
| 32   | 3    | 2    | 2 (=log2(32)/2)  | 1    | 38    |
| 64   | 3    | 2    | 3 (=log2(64)/2)  | 1    | 42    |
| 128  | 3    | 2    | 3 (=log2(128)/2) | 1    | 42    |

dg_i formula: `floor(log2(N) / 2)`.

**Reference**: `test-median.cpp` main(), lines 400–403.

### Reference parameters (multi-CT f,g, `testMedianMultiCtxtAdv`)

L=256 blocks. Depth formula:

```
multiplicativeDepth = 4*(dg_c + df_c + dg_i + df_i) + 3 + 2 + 3
```

The `+3+2+3` = `+8` overhead (vs `+6` for single-CT) accounts for multi-CT
accumulation (+2) and median-specific masking (+3 vs +0 for non-TC).

### HEonGPU implementation (`20_ckks_median.cpp`)

Single-CT f,g at `n=131072`, `scale=59`.

| N   | dg_i | depth | Q_size | dnum |
|----:|-----:|------:|-------:|-----:|
| 8   | 1    | 34    | 35     | 2    |
| 16  | 2    | 38    | 39     | 2    |
| 32  | 2    | 38    | 39     | 2    |
| 64  | 3    | 42    | 43     | 3    |
| 128 | 3    | 42    | 43     | 3    |

The depth formula `4*(dg_c+df_c+dg_i+df_i)+6`, `dg_c=3`, `df_c=2`, `df_i=1`
all match the reference exactly.

The `+6` overhead includes:
- +1 TransR mask
- +3 correction term (ct*ct, *4, *triMask)
- +1 indicator input scaling
- +1 indicator output multiply

---

## 5. 21\_ckks\_ranking\_multi

**Algorithm**: Multi-ciphertext ranking (Algorithm 7, complementary
optimization) with optional tie correction (Algorithm 6).

**Reference**: `test-ranking.cpp`, functions `testRankingMultiCtxt` (Chebyshev)
and `testRankingMultiCtxtAdv` (f,g).

### Dispatch logic (`test-ranking.cpp` main, lines 334–479)

The reference uses two distinct comparison methods depending on N:

| N range  | Mode  | Compare method | Reference function |
|---------:|:------|:---------------|:-------------------|
| N <= 256 | basic | Chebyshev      | `testRankingMultiCtxt` |
| N <= 256 | TC    | Chebyshev      | `testRankingMultiCtxt` |
| N > 256  | basic | f,g (dg=3,df=2)| `testRankingMultiCtxtAdv` |
| N > 256  | TC    | f,g (dg=3,df=2)| `testRankingMultiCtxtAdv` |

### Reference parameters: Chebyshev path (N <= 256)

`testRankingMultiCtxt` (line 106):

```
integralPrecision  = 1
decimalPrecision   = 35
multiplicativeDepth = compareDepth + 2 + (tieCorrection ? 3 : 0)
numSlots           = 128 * 128 = 16384
L (subVectorLength) = 128
```

**compareDepth selection** (separate tables for basic and TC):

Basic (no tie correction):
| N range  | compareDepth | degree (via depth2degree) | multiplicativeDepth |
|---------:|:------------:|:------------------------:|:-------------------:|
| N <= 8   | 7            | 59                       | 9                   |
| N <= 16  | 8            | 119                      | 10                  |
| N <= 64  | 10           | 495                      | 12                  |
| N <= 256 | 11           | 1007                     | 13                  |

TC (tie correction):
| N range  | compareDepth | degree (via depth2degree) | multiplicativeDepth |
|---------:|:------------:|:------------------------:|:-------------------:|
| N <= 8   | 7            | 59                       | 12                  |
| N <= 16  | 9            | 247                      | 14                  |
| N <= 64  | 10           | 495                      | 15                  |
| N <= 256 | 12           | 2031                     | 17                  |

**Reference**: `test-ranking.cpp` main(), lines 340–344 (basic) and
lines 413–417 (TC).

The `+2` in the depth formula accounts for multi-CT overhead (TransR mask +
maskColumn0). The `+3` for TC accounts for sign^2 (1), mask*E (1), and an
additional masking step (1).

### Reference parameters: f,g path (N > 256)

`testRankingMultiCtxtAdv` (line 212):

```
integralPrecision  = 1
decimalPrecision   = 45
multiplicativeDepth = 4*(dg + df) + 3 + 1 + (tieCorrection ? 3 : 0)
                    = 4*(3 + 2) + 3 + 1 + (TC ? 3 : 0)
                    = 24 (basic) / 27 (TC)
numSlots           = 128 * 128 = 16384
L (subVectorLength) = 128
dg = 3, df = 2
```

The `4*(dg+df)` = 20 levels for five degree-7 polynomial evaluations (4 levels
each in OpenFHE's Paterson-Stockmeyer). The `+3+1` = 4 levels of overhead for
the multi-CT accumulation framework (TransR mask, maskColumn0, complement
accumulation, and final summation step).

**Reference**: `test-ranking.cpp`, lines 385–386 (dg/df) and lines 458–459
(same for TC path).

### HEonGPU implementation (`21_ckks_ranking_multi.cpp`)

All modes use `n=131072` (3500-bit security budget) with `dnum=1`.

| Mode            | Cheby degree | depth | Q chain           | P chain    | scale |
|:----------------|:-------------|------:|:------------------|:-----------|------:|
| Basic Cheby     | 1007         | 13    | {60, 45x13}       | {60x47}    | 45    |
| TC Cheby        | 2031         | 17    | {60, 45x17}       | {60x44}    | 45    |
| Basic f,g       | —            | 24    | {60, 45x24}       | {60x39}    | 45    |
| TC f,g          | —            | 27    | {60, 45x27}       | {60x37}    | 45    |

**Key parameter mappings from reference**:
- Chebyshev degrees: 1007 = `depth2degree(11)`, 2031 = `depth2degree(12)`
- Chebyshev depth 13: `compareDepth(11) + 2` (basic), 17: `compareDepth(12) + 2 + 3` (TC)
- f,g depth 24: `4*(3+2) + 3 + 1` (basic), 27: `24 + 3` (TC)
- f,g dg=3, df=2: identical to reference

**Differences from reference**:
- Scale: 45-bit (vs reference's 35-bit for Chebyshev, 45-bit for f,g)
- FirstModSize: 60-bit (vs reference's 36-bit for Chebyshev, 46-bit for f,g)
- These differences compensate for fixed-scale vs FLEXIBLEAUTO (see Section 9)

---

## 6. 22\_ckks\_sorting\_paper

**Algorithm**: Sorting with tie correction (Algorithm 5 + Algorithm 6).
Always uses f,g composition.

**Reference**: `test-sorting.cpp`, functions `testSortingAdv` (single-CT)
and `testSortingMultiCtxtAdv` (multi-CT).

### Dispatch logic (`test-sorting.cpp` main, lines 360–419)

The reference always uses f,g composition with L=256 and tie correction=true.

### Reference parameters: single-CT (`testSortingAdv`)

```
integralPrecision  = 1
decimalPrecision   = 59
multiplicativeDepth = 4*(dg_c + df_c + dg_i + df_i) + 4 + (tieCorrection ? 3 : 0)
                    = 4*(dg_c + df_c + dg_i + df_i) + 7   (TC always on)
dg_c = 3, df_c = 2
dg_i = (log2(N) + 1) / 2   (integer division)
df_i = 2
```

| N   | dg_i                  | depth (4*sum+7) |
|----:|:---------------------:|:---------------:|
| 4   | 1 (=(2+1)/2)          | 39              |
| 8   | 2 (=(3+1)/2)          | 43              |
| 16  | 2 (=(4+1)/2)          | 43              |
| 32  | 3 (=(5+1)/2)          | 47              |
| 64  | 3 (=(6+1)/2)          | 47              |
| 128 | 4 (=(7+1)/2)          | 51              |
| 256 | 4 (=(8+1)/2)          | 51              |

**Reference**: `test-sorting.cpp` main(), lines 376–379.

The `+4` overhead (single-CT) comes from:
- +1 TransR mask (multiply_plain + rescale)
- +1 indicator × rank (multiply + rescale)
- +1 sumR folding product
- +1 final extraction

The `+3` for TC: sign^2 (1), mask*E multiply (1), TC accumulation (1).

### Reference parameters: multi-CT (`testSortingMultiCtxtAdv`)

```
multiplicativeDepth = 4*(dg_c + df_c + dg_i + df_i) + 6 + (tieCorrection ? 3 : 0)
                    = 4*(dg_c + df_c + dg_i + df_i) + 9   (TC always on)
```

Multi-CT adds `+2` over single-CT for the extra maskColumn0 and
transposeColumn operations in the block accumulation phase.

### HEonGPU implementation (`22_ckks_sorting_paper.cpp`)

Single-CT f,g at `n=131072`. Uses adaptive scale selection to match the
paper's dnum for each N (since FLEXIBLEAUTO achieves lower effective Q-bits):

```
dg_c = 3, df_c = 2
dg_i = (log2(N) + 1) / 2
df_i = 2
depth = 4*(dg_c + df_c + dg_i + df_i) + 7
```

The implementation finds the largest `scale_bits` (starting from 59) such that
`dnum <= paper_dnum[N]`:

| N   | dg_i | depth | scale | dnum | paper dnum |
|----:|:----:|------:|------:|-----:|-----------:|
| 4   | 1    | 39    | 57    | 2    | 2          |
| 8   | 2    | 43    | 59    | 3    | 3          |
| 16  | 2    | 43    | 59    | 3    | 3          |
| 32  | 3    | 47    | 57    | 4    | 3          |
| 64  | 3    | 47    | 57    | 4    | 4          |
| 128 | 4    | 51    | 54    | 5    | 5          |
| 256 | 4    | 51    | 54    | 5    | 5          |

**Key parameter mappings from reference**:
- `dg_c=3`, `df_c=2`, `df_i=2`: identical to reference
- `dg_i = (log2(N)+1)/2`: identical to reference
- `depth = 4*sum + 7`: matches reference's `4*sum + 4 + 3` (TC always on)
- `decimalPrecision=59`: matched (via adaptive scale ≈ 54–59)

---

## 7. 23\_ckks\_ranking\_tie\_correction

**Algorithm**: Single-ciphertext ranking (Algorithm 3) with optional tie
correction (Algorithm 6). Paper-spec parameters at `n=32768`.

**Reference**: `test-ranking.cpp`, function `testRanking`.

### Reference parameters (single-CT Chebyshev, `testRanking`)

```
integralPrecision  = 1
decimalPrecision   = (N <= 16 && !tieCorrection) ? 30 : 35
multiplicativeDepth = compareDepth + 1 + (tieCorrection ? 3 : 0)
numSlots           = N * N
```

Note: single-CT uses `+1` overhead (vs `+2` for multi-CT), because there is
no maskColumn0 step — only the TransR mask is needed.

**compareDepth selection** (same table for basic; different for TC):

Basic (no tie correction):
| N range  | compareDepth | degree | decimalPrecision | depth |
|---------:|:------------:|-------:|:----------------:|------:|
| N <= 8   | 7            | 59     | 30               | 8     |
| N <= 16  | 8            | 119    | 30               | 9     |
| N <= 32  | 9            | 247    | 35               | 10    |
| N <= 64  | 10           | 495    | 35               | 11    |
| N <= 128 | 11           | 1007   | 35               | 12    |

TC (tie correction):
| N range  | compareDepth | degree | decimalPrecision | depth |
|---------:|:------------:|-------:|:----------------:|------:|
| N <= 8   | 7            | 59     | 35               | 11    |
| N <= 16  | 9            | 247    | 35               | 13    |
| N <= 32  | 9            | 247    | 35               | 13    |
| N <= 64  | 10           | 495    | 35               | 14    |
| N <= 128 | 11           | 1007   | 35               | 15    |

**Reference**: `test-ranking.cpp` main(), lines 340–344 (basic) and
lines 413–417 (TC). `testRanking` line 18 (decimalPrecision selection).

### HEonGPU implementation (`23_ckks_ranking_tie_correction.cpp`)

Fixed `n=32768` (881-bit budget). Two parameter tiers:

**Tier 1 (N <= 32)**:
- Q = {36, 35x14} = 15 primes
- P = {36x8}
- scale = 35, dnum = 2
- Matches reference's `decimalPrecision=35`, `firstModSize=36`

**Tier 2 (N > 32)**:
- Q = {60, 45x14} = 15 primes
- P = {60x3}
- scale = 45, dnum = 5
- Uses larger scale for fixed-scale CKKS precision (see Section 9)

Available depth: 14. This limits:
- Basic: up to N=128 (depth 13, compareDepth=11 via `ceil(log2(2048))=11`, +2)
- TC: up to N=64 (depth 14, compareDepth=10 via `ceil(log2(1024))=10`, +4)

The Chebyshev degree selection **intentionally** uses higher degrees than the
reference's `depth2degree` values:

| N    | Reference degree | Our degree | Ratio |
|-----:|-----------------:|-----------:|------:|
| <=8  | 59               | 127        | 2.2x  |
| <=16 | 119              | 255        | 2.1x  |
| <=32 | 247              | 511        | 2.1x  |
| <=64 | 495              | 1023       | 2.1x  |
| <=128| 1007             | 2047       | 2.0x  |

This is a deliberate compensation for fixed-scale CKKS: without FLEXIBLEAUTO's
dynamic noise management, each level accumulates more noise, so the sign
approximation must be higher-quality (higher degree) to still round correctly
after decryption. The roughly 2x degree increase costs ~1 extra level in
HEonGPU's BSGS evaluation, which fits within the available depth budget.

---

## 8. 24\_ckks\_ranking\_tie\_correction\_extended

**Algorithm**: Same as 23\_ but extended beyond paper spec using larger ring
dimensions. This is a thesis contribution (Section 1.2.5) showing that GPU
parallelism enables larger CKKS parameters practically.

**Reference**: Same as 23\_, but the extension to larger N and ring dimensions
has no direct reference counterpart.

### HEonGPU parameter tiers

| N range / mode         | n       | Q chain        | P chain   | scale | dnum | depth |
|:-----------------------|--------:|:---------------|:----------|------:|-----:|------:|
| N<=32 basic            | 32768   | {36, 35x14}    | {36x8}    | 35    | 2    | <=14  |
| N<=128 basic (large N) | 32768   | {60, 45x14}    | {60x3}    | 45    | 5    | <=14  |
| N=128 TC               | 65536   | {60, 45x15}    | {60x13}   | 45    | 2    | 15    |
| N=256                  | 131072  | {60, 45x15}    | {60x13}   | 45    | 2    | <=15  |
| N=512                  | 524288  | {60, 45x15}    | {60x13}   | 45    | 2    | <=15  |

The first two tiers are identical to 23\_ (paper-spec). The extended tiers
use progressively larger ring dimensions to accommodate depth-15 circuits
that don't fit at n=32768.

Chebyshev degree selection is the same as 23\_ (`selectChebyshevDegree()`).

---

## 9. Key Differences: OpenFHE vs HEonGPU

### FLEXIBLEAUTO vs Fixed-Scale CKKS

This is the **single most important** difference between the reference
implementation and all HEonGPU examples.

**OpenFHE (FLEXIBLEAUTO)**: After each multiplication, the library dynamically
adjusts the scale of intermediate ciphertexts. This means:
- Each modulus prime can have a different size
- Scales after rescaling are not exactly `2^s` but tracked precisely
- Noise growth is managed adaptively, giving better precision per level

**HEonGPU (Fixed-Scale)**: Every level uses the same scale `2^s`, and every
Q prime (except the first) has size exactly `s` bits. This means:
- Predictable but suboptimal noise management
- Each multiplication adds a fixed amount of noise
- Deep circuits (like f,g with 5 sequential degree-7 evaluations)
  accumulate more noise than FLEXIBLEAUTO

### Compensating strategies

Without FLEXIBLEAUTO, fixed-scale CKKS has worse per-level precision. All
parameter deviations from the reference are **intentional compensations**:

| Parameter | Reference (FLEXIBLEAUTO) | HEonGPU (Fixed-Scale) | Why we deviate |
|---|---|---|---|
| Chebyshev scale | 35-bit | 45-bit | +10 bits/level counteracts fixed noise growth |
| f,g scale | 45-bit | 45-bit | Matched (already high in reference) |
| FirstModSize | 36 or 46 | 60-bit | More initial precision headroom before first rescale |
| Ring dimension | Auto (32768–65536) | Often 131072 | More Q+P budget = lower dnum = less key-switching noise |
| Chebyshev degrees | depth2degree (59, 247, 1007, 2031) | ~2x higher (127, 511, 2047) | Better sign approximation absorbs noisier intermediate values |

The 21\_ multi-CT ranking is an exception: it uses the reference's exact
degrees (1007/2031) because n=131072 with dnum=1 already provides enough
precision headroom that degree compensation is not needed.

### f,g accuracy gap

The f,g composition path (N > 256 ranking, all sorting, median) is where the
FLEXIBLEAUTO advantage matters most. Five sequential degree-7 evaluations
compound noise multiplicatively. In fixed-scale CKKS, this results in
measurably higher output error than the reference, which may cause rank
mismatches even with optimal parameter choices.

For the Chebyshev path (N <= 256 ranking, minimum), the accuracy gap is
smaller because there is only a single polynomial evaluation, and the noise
is dominated by the Chebyshev approximation quality rather than accumulated
CKKS noise.

### Polynomial evaluation method

**OpenFHE**: Uses `EvalPolyLinear` (Horner's method in monomial basis) for
degree-7 f,g polynomials. Uses Paterson-Stockmeyer (tree-based) for
high-degree Chebyshev.

**HEonGPU**: Uses BSGS Chebyshev evaluation for both f,g and sign
approximation. The f,g polynomials are first sampled at Chebyshev nodes
and converted to Chebyshev coefficients, then evaluated via BSGS. For
degree-7 polynomials, both monomial and Chebyshev representations span the
same polynomial space, so the mathematical result is identical — but the
intermediate noise profile differs.

### `lead` parameter in Chebyshev evaluation

HEonGPU's `evaluate_poly` has a `lead` parameter that controls whether
the leading coefficient is treated specially (multiplied in at the last step
rather than accumulated into the tree). This affects noise:

- `lead=true`: Used for f,g functions (essential for accuracy — switching
  to `lead=false` causes catastrophic errors in the f,g path)
- `lead=false`: Used for Chebyshev sign approximation in single-CT ranking
  (23\_ and 24\_)
- The multi-CT ranking (21\_) uses `lead=false` for Chebyshev sign and
  `lead=true` for f,g, matching each method's requirements
