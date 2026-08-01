# Flank-400 reliability diagrams

Figures:
- `figures/logit_based_flank400/reliability_multiclass_flank400_methods.png`
- `figures/logit_based_flank400/reliability_acceptor_flank400_methods.png`
- `figures/logit_based_flank400/reliability_donor_flank400_methods.png`

Tables:
- `tables/logit_based_flank400/reliability_bins_flank400_methods.csv`
- `tables/logit_based_flank400/reliability_summary_flank400_methods.csv`

Methods shown:
- Uncalibrated
- True-logit genome-weighted vector T
- OpenSpliceAI-style vector T

Summary:
Uncalibrated weighted multiclass ECE was 0.00163185. True-logit genome-weighted vector T reduced this to 0.0000289988. OpenSpliceAI-style vector T further reduced the point-estimate weighted multiclass ECE to 0.0000116824.

Class-wise ECE:
- Uncalibrated acceptor/donor: 0.00075937 / 0.00087856
- True-logit genome-weighted vector T acceptor/donor: 0.00001962 / 0.00001745
- OpenSpliceAI-style vector T acceptor/donor: 0.00000999 / 0.00001406

Interpretation:
The reliability diagrams provide visual support for the quantitative calibration results. In flank-400, vector-temperature scaling strongly improves weighted calibration under the genome-position prior, while detection AUPRC remains essentially unchanged.
