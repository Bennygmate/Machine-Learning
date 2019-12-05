Tester:     weka.experiment.PairedCorrectedTTester -G 4,5,6 -D 1 -R 2 -S 0.05 -V -result-matrix "weka.experiment.ResultMatrixPlainText -mean-prec 2 -stddev-prec 2 -col-name-width 0 -row-name-width 25 -mean-width 0 -stddev-width 0 -sig-width 0 -count-width 5 -show-stddev -print-col-names -print-row-names -enum-col-names"
Analysing:  Percent_incorrect
Datasets:   3
Resultsets: 2
Confidence: 0.05 (two tailed)
Sorted by:  -
Date:       08/04/17 16:06


Dataset                   (1) rules.ZeroR '' | (2) trees.J48 '
--------------------------------------------------------------
anneal                   (100)   23.83(0.55) |    1.43(1.04) *
audiology                (100)   74.79(1.85) |   22.74(7.47) *
vote                     (100)   38.62(0.81) |    3.43(2.56) *
--------------------------------------------------------------
                                     (v/ /*) |         (0/0/3)


Key:
(1) rules.ZeroR '' 48055541465867954
(2) trees.J48 '-C 0.25 -M 2' -217733168393644444

