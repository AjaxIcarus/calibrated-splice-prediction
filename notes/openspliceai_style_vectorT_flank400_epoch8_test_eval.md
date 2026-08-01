# OpenSpliceAI-style vector-temperature test evaluation: flank-400 epoch-8

Model:
`results/best_models/flank400_focal_best.pt`

Calibration:
`results/openspliceai_style_vectorT_flank400_epoch8_fullval_3000batched/temperature_best.txt`

Temperature:
`[0.3837826, 0.36834455, 0.38703716]`

Calibration fitting:
- Full validation H5/model-output calibration
- Optimized through epoch 1800
- NLL plateaued by approximately epoch 400
- Best NLL: 0.00026906293351203203

Test cache:
`results/logit_cache_flank400_epoch8/test_sampled_logits.npz`

Test summary:
- Weighted multiclass ECE: 0.0000116824
- Weighted NLL: 0.0002578195
- Acceptor ECE: 0.0000099924
- Donor ECE: 0.0000140593
- Acceptor AUPRC: 0.9994644414
- Donor AUPRC: 0.9995892897
- Total positions seen: 416,160,000
- Positives seen: 106,088
- Sampled negatives: 500,000
- Negative weight: 832.107824

Threshold results:
- Acceptor @0.01: precision 0.996984, recall 0.972692
- Donor @0.01: precision 0.997384, recall 0.977047
- Acceptor @0.05: precision 0.999034, recall 0.936368
- Donor @0.05: precision 0.998898, recall 0.939790

Interpretation:
The converged OpenSpliceAI-style class-wise vector-temperature baseline gives excellent weighted calibration on the flank-400 epoch-8 model while preserving splice-site detection performance.
