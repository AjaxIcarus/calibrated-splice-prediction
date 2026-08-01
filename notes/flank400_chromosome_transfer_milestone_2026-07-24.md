# Flank-400 chromosome-transfer milestone

Frozen: 2026-07-24

## Authoritative models

- Primary: seed 11, epoch 11
- Replication: seed 23, epoch 14
- Epoch 8 is historical and excluded.

## Evaluation scope

Chromosomes: chr1, chr3, chr5, chr7, chr9

Each model was evaluated over 416,140,000 positions using the same
aggregation and pooled-ECE implementation.

## Pooled results

| Model | Method | ECE | NLL |
|---|---|---:|---:|
| seed11_epoch11 | Uncalibrated | 0.001507295858 | 0.001698358538 |
| seed11_epoch11 | Fixed global T=1.1 | 0.002519030113 | 0.002716969665 |
| seed11_epoch11 | Unweighted vector-T | 0.000046921955 | 0.002062224107 |
| seed11_epoch11 | Genome-weighted vector-T | 0.000020870360 | 0.000252214137 |
| seed11_epoch11 | OSAI-style vector-T | 0.000004553865 | 0.000244193205 |
| seed23_epoch14 | Uncalibrated | 0.001544696386 | 0.001725679388 |
| seed23_epoch14 | Fixed global T=1.1 | 0.002586726680 | 0.002775464228 |
| seed23_epoch14 | Unweighted vector-T | 0.000050493954 | 0.001974532575 |
| seed23_epoch14 | Genome-weighted vector-T | 0.000027388731 | 0.000240350851 |
| seed23_epoch14 | OSAI-style vector-T | 0.000003084788 | 0.000230533274 |

## Frozen conclusion

The qualitative result replicated across both models. OSAI-style
class-wise vector temperature scaling was best on pooled ECE and NLL.
Genome-weighted vector-T also improved both metrics. Unweighted
vector-T improved ECE but worsened NLL, while fixed T=1.1 worsened
both metrics.

This is a five-chromosome transfer result, not a claim of complete
genome-wide evaluation or an inferential chromosome-level test.
