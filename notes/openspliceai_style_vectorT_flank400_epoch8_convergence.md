# OpenSpliceAI-style vector-temperature calibration: flank-400 epoch-8

Model:
`results/best_models/flank400_focal_best.pt`

Calibration dataset:
`data/processed_h5_flank400/dataset_validation.h5`

Output directory:
`results/openspliceai_style_vectorT_flank400_epoch8_fullval_3000batched`

Optimization:
- OpenSpliceAI-style class-wise vector temperature
- Full validation H5/model-output calibration
- CPU batched run with checkpoint/resume
- Optimized through epoch 1800

Convergence:
- NLL plateaued by approximately epoch 400.
- Best NLL remained stable through epoch 1800.
- Best NLL: 0.00026906293351203203
- Best temperature: [0.3837826, 0.36834455, 0.38703716]

Interpretation:
Further optimization to 3000 epochs was not necessary because the validation NLL and temperature vector had stabilized. The test evaluation uses the saved `temperature_best.txt`.
