# PyHTK

HTK features in Python

This project contains a Python implementation of the MFCC features as computed by HTK.

## What is HTK?

HTK is a respected toolkit used mainly by the speech community to perform research in speech recognition. Although quite old, many newer systems emulate the same feature extraction pipeline as used in HTK. Besides being thouroughly tested it is also well documented in a manual known as the HTK Book.

You can find more information about HTK on their official website: http://htk.eng.cam.ac.uk/

## What is MFCC?

The MFCC feature set, as implemented in this little project, is one of the best performing techniques for modeling speech in tasks like speech recognition. While there may be others that are margianlly better in specific cases, MFCCs remain as a strong baseline for many standard benchmarks.

## How to use?

Simply copy the HTKFeat.py file to your project and use the MFCC_HTK class from within. The class is throughly documented.

The HTK.py file isn't neccessary, but if you want, you can use it to compute the same features using the HCopy program which is a part of HTK.

For a demonstration of its use, take a look at the contents of the python-test directory.

## How does it work?

Open the HTKFeaturesExplained notebook in the python-notebooks folder and play around with it. Everythin is explained there.

## Who made this?

If you have any questions, feel free to contact me at: danijel@pja.edu.pl