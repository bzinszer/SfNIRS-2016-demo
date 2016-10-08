# SfNIRS-2016-demo
Software for performing representational similarity analysis (RSA)-based decoding on fNIRS data. Demonstrated by Zinszer, Bayet, Emberson, &amp; Aslin (2016) at Society for fNIRS meeting (October 2016).

To run the analyses in the poster, open the demo script:

SfNIRS_8class_decode_demo_15Oct2016.m (commentary provided within)

The demo script relies on several support functions included in this repository. fNIRS data are also included in the repository (or will be very soon), but you should be able to run this script over your own data with minimal modifications.

Permutation-based significance testing is contained in a separate script that must be run after completeing the above demo (with all variables still in the Matlab workspace):

null_distribution_generator_15Oct2016.m

--------------------------------------------

Please see Emberson, Zinszer, Raizada, &amp; Aslin (2016) regarding Multichannel Pattern Analysis for fNIRS.

- bioRxiv: http://biorxiv.org/content/early/2016/06/30/061234

- Github: http://teammcpa.github.io/EmbersonZinszerMCPA/

Please see Anderson, Zinszer, &amp; Raizada (2015) regarding representaitonal similarity based decoding and encoding methods for fMRI, the foundation upon which this work was built.

- paywall access via NeuroImage journal: http://www.sciencedirect.com/science/article/pii/S1053811915011489

- free access via Raizada Lab at University of Rochester: http://www.bcs.rochester.edu/people/raizada/papers/AndersonZinszerRaizada_RepSimEncoding_NeuroImage2016.pdf
