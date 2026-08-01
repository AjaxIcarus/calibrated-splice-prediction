# Flank-80 vs flank-400 calibration and detection comparison

Table:
`tables/flank80_vs_flank400_calibration_detection_rounded.csv`

Main result:
The flank-400 epoch-8 model is substantially stronger than the flank-80 epoch-2 model in splice-site detection, while the same calibration story holds.

Detection:
- Flank-80 uncalibrated acceptor AUPRC: 0.990909
- Flank-80 uncalibrated donor AUPRC: 0.994971
- Flank-400 uncalibrated acceptor AUPRC: 0.999464
- Flank-400 uncalibrated donor AUPRC: 0.999590

Calibration:
- Flank-80 uncalibrated weighted ECE: 0.00560995
- Flank-80 OpenSpliceAI-style vector T weighted ECE: 0.0000112525
- Flank-400 uncalibrated weighted ECE: 0.00163185
- Flank-400 OpenSpliceAI-style vector T weighted ECE: 0.0000116824

Weighted NLL:
- Flank-80 uncalibrated weighted NLL: 0.00623967
- Flank-80 OpenSpliceAI-style vector T weighted NLL: 0.000804295
- Flank-400 uncalibrated weighted NLL: 0.00183797
- Flank-400 OpenSpliceAI-style vector T weighted NLL: 0.00025782

Interpretation:
The stronger flank-400 model improves detection and raw calibration, but post-hoc vector-temperature calibration remains useful. OpenSpliceAI-style vector T gives the best or near-best weighted calibration metrics in both settings while preserving acceptor/donor AUPRC. This supports the paper's central claim that splice-site detection and calibrated probability estimation are distinct objectives under extreme per-nucleotide class imbalance.
