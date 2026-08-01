# True-logit Bootstrap Summary

Bootstrap replicates: 200

Negative sample weight: 832.107824

| Method                         | Metric                  |     Mean | 95% CI               |
|:-------------------------------|:------------------------|---------:|:---------------------|
| Uncalibrated                   | Weighted multiclass ECE | 0.001633 | [0.001613, 0.001654] |
| Uncalibrated                   | Weighted multiclass NLL | 0.00184  | [0.001815, 0.001866] |
| Uncalibrated                   | Acceptor AUPRC          | 0.999457 | [0.999372, 0.999543] |
| Uncalibrated                   | Donor AUPRC             | 0.999583 | [0.999499, 0.999662] |
| Global T=1.1                   | Weighted multiclass ECE | 0.002696 | [0.002671, 0.002718] |
| Global T=1.1                   | Weighted multiclass NLL | 0.002911 | [0.002883, 0.002942] |
| Global T=1.1                   | Acceptor AUPRC          | 0.999458 | [0.999373, 0.999543] |
| Global T=1.1                   | Donor AUPRC             | 0.999583 | [0.999499, 0.999662] |
| Logit unweighted vector T      | Weighted multiclass ECE | 1.9e-05  | [0.000010, 0.000030] |
| Logit unweighted vector T      | Weighted multiclass NLL | 0.00026  | [0.000238, 0.000284] |
| Logit unweighted vector T      | Acceptor AUPRC          | 0.999458 | [0.999373, 0.999543] |
| Logit unweighted vector T      | Donor AUPRC             | 0.999582 | [0.999498, 0.999662] |
| Logit genome-weighted vector T | Weighted multiclass ECE | 1.9e-05  | [0.000010, 0.000030] |
| Logit genome-weighted vector T | Weighted multiclass NLL | 0.00026  | [0.000238, 0.000284] |
| Logit genome-weighted vector T | Acceptor AUPRC          | 0.999458 | [0.999373, 0.999543] |
| Logit genome-weighted vector T | Donor AUPRC             | 0.999582 | [0.999498, 0.999662] |
