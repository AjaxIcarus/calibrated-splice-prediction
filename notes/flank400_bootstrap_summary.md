# Flank-400 bootstrap summary

Table:
`tables/flank400_bootstrap_summary_for_paper_rounded.csv`

Bootstrap:
- n = 200
- sampled flank-400 test logit cache
- weighted by genome-position negative prior
- metrics: weighted multiclass ECE, weighted NLL, acceptor AUPRC, donor AUPRC

Main result:
OpenSpliceAI-style vector T achieved the lowest point-estimate weighted ECE/NLL on flank-400, with bootstrap intervals overlapping the genome-prior weighted true-logit vector T.

Key values:

Uncalibrated:
- ECE: 0.00163345 [0.00161308, 0.00165384]
- NLL: 0.00183979 [0.00181491, 0.00186598]

True-logit genome-weighted vector T:
- ECE: 0.0000346041 [0.0000255569, 0.0000441710]
- NLL: 0.000269186 [0.000252611, 0.000288178]
- Acceptor AUPRC: 0.999457 [0.999372, 0.999543]
- Donor AUPRC: 0.999581 [0.999497, 0.999661]

OpenSpliceAI-style vector T:
- ECE: 0.00001893 [0.00000977, 0.00003010]
- NLL: 0.00025968 [0.00023820, 0.00028369]
- Acceptor AUPRC: 0.999458 [0.999373, 0.999543]
- Donor AUPRC: 0.999582 [0.999498, 0.999662]

Interpretation:
The stronger flank-400 model greatly improves detection compared with flank-80, but calibration still matters. Post-hoc vector-temperature scaling sharply reduces weighted ECE/NLL while leaving ranking-based detection essentially unchanged.
