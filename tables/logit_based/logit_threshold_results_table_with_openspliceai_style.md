# True-logit Threshold Results Table with OpenSpliceAI-style Baseline

| Method                                 | Class    |   Threshold |   Precision |   Recall |   Predicted positives |   True positives |
|:---------------------------------------|:---------|------------:|------------:|---------:|----------------------:|-----------------:|
| Uncalibrated                           | Acceptor |        0.01 |      0.7032 |   0.9992 |                 75348 |            52982 |
| Uncalibrated                           | Acceptor |        0.05 |      0.935  |   0.9844 |                 55824 |            52195 |
| Uncalibrated                           | Acceptor |        0.1  |      0.9746 |   0.9226 |                 50195 |            48918 |
| Uncalibrated                           | Acceptor |        0.5  |      1      |   0.075  |                  3978 |             3978 |
| Uncalibrated                           | Donor    |        0.01 |      0.5988 |   0.9991 |                 88536 |            53017 |
| Uncalibrated                           | Donor    |        0.05 |      0.9489 |   0.9922 |                 55487 |            52652 |
| Uncalibrated                           | Donor    |        0.1  |      0.981  |   0.9589 |                 51872 |            50885 |
| Uncalibrated                           | Donor    |        0.5  |      0.9999 |   0.1432 |                  7601 |             7600 |
| Logit genome-weighted vector T         | Acceptor |        0.01 |      0.9801 |   0.8954 |                 48441 |            47477 |
| Logit genome-weighted vector T         | Acceptor |        0.05 |      0.9935 |   0.6619 |                 35327 |            35097 |
| Logit genome-weighted vector T         | Acceptor |        0.1  |      0.9962 |   0.5022 |                 26728 |            26627 |
| Logit genome-weighted vector T         | Acceptor |        0.5  |      0.9994 |   0.0966 |                  5127 |             5124 |
| Logit genome-weighted vector T         | Donor    |        0.01 |      0.9852 |   0.9403 |                 50644 |            49894 |
| Logit genome-weighted vector T         | Donor    |        0.05 |      0.9951 |   0.7542 |                 40220 |            40022 |
| Logit genome-weighted vector T         | Donor    |        0.1  |      0.9974 |   0.6063 |                 32257 |            32173 |
| Logit genome-weighted vector T         | Donor    |        0.5  |      0.9998 |   0.1585 |                  8415 |             8413 |
| OpenSpliceAI-style unweighted vector T | Acceptor |        0.01 |      0.983  |   0.8684 |                 46838 |            46044 |
| OpenSpliceAI-style unweighted vector T | Acceptor |        0.05 |      0.9938 |   0.6491 |                 34633 |            34420 |
| OpenSpliceAI-style unweighted vector T | Acceptor |        0.1  |      0.9962 |   0.5104 |                 27165 |            27063 |
| OpenSpliceAI-style unweighted vector T | Acceptor |        0.5  |      0.9994 |   0.137  |                  7267 |             7263 |
| OpenSpliceAI-style unweighted vector T | Donor    |        0.01 |      0.987  |   0.9236 |                 49656 |            49009 |
| OpenSpliceAI-style unweighted vector T | Donor    |        0.05 |      0.9953 |   0.7465 |                 39799 |            39610 |
| OpenSpliceAI-style unweighted vector T | Donor    |        0.1  |      0.9972 |   0.6182 |                 32894 |            32802 |
| OpenSpliceAI-style unweighted vector T | Donor    |        0.5  |      0.9995 |   0.2092 |                 11109 |            11103 |
