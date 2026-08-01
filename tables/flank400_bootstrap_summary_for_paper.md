# Flank-400 bootstrap summary

| method                              | metric                  |        mean |         std |   ci_lower_2.5 |   ci_upper_97.5 |
|:------------------------------------|:------------------------|------------:|------------:|---------------:|----------------:|
| uncalibrated                        | weighted_multiclass_ece | 0.00163345  | 1.14851e-05 |    0.00161308  |     0.00165384  |
| global T=1.1                        | weighted_multiclass_ece | 0.00269584  | 1.32881e-05 |    0.00267082  |     0.00271832  |
| true-logit unweighted vector T      | weighted_multiclass_ece | 0.000130709 | 2.91245e-05 |    8.02172e-05 |     0.000187862 |
| true-logit genome-weighted vector T | weighted_multiclass_ece | 3.46041e-05 | 4.84979e-06 |    2.55569e-05 |     4.4171e-05  |
| OpenSpliceAI-style vector T         | weighted_multiclass_ece | 1.893e-05   | 5.39456e-06 |    9.77253e-06 |     3.00974e-05 |
| uncalibrated                        | weighted_multiclass_nll | 0.00183979  | 1.38105e-05 |    0.00181491  |     0.00186598  |
| global T=1.1                        | weighted_multiclass_nll | 0.00291111  | 1.54677e-05 |    0.00288278  |     0.00294157  |
| true-logit unweighted vector T      | weighted_multiclass_nll | 0.00229948  | 9.72525e-05 |    0.00212852  |     0.0024722   |
| true-logit genome-weighted vector T | weighted_multiclass_nll | 0.000269186 | 9.70832e-06 |    0.000252611 |     0.000288178 |
| OpenSpliceAI-style vector T         | weighted_multiclass_nll | 0.00025968  | 1.23781e-05 |    0.000238204 |     0.000283691 |
| uncalibrated                        | acceptor_auprc          | 0.999457    | 4.24482e-05 |    0.999372    |     0.999543    |
| global T=1.1                        | acceptor_auprc          | 0.999458    | 4.24332e-05 |    0.999373    |     0.999543    |
| true-logit unweighted vector T      | acceptor_auprc          | 0.99941     | 4.35456e-05 |    0.999321    |     0.999488    |
| true-logit genome-weighted vector T | acceptor_auprc          | 0.999457    | 4.25102e-05 |    0.999372    |     0.999543    |
| OpenSpliceAI-style vector T         | acceptor_auprc          | 0.999458    | 4.24238e-05 |    0.999373    |     0.999543    |
| uncalibrated                        | donor_auprc             | 0.999583    | 4.1166e-05  |    0.999499    |     0.999662    |
| global T=1.1                        | donor_auprc             | 0.999583    | 4.11543e-05 |    0.999499    |     0.999662    |
| true-logit unweighted vector T      | donor_auprc             | 0.999541    | 4.11964e-05 |    0.99945     |     0.999616    |
| true-logit genome-weighted vector T | donor_auprc             | 0.999581    | 4.15433e-05 |    0.999497    |     0.999661    |
| OpenSpliceAI-style vector T         | donor_auprc             | 0.999582    | 4.12956e-05 |    0.999498    |     0.999662    |
