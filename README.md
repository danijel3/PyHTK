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

If you prefer, you can also install the file as a library:

```bash
pip install https://github.com/danijel3/PyHTK.git
```

Then you can access the class in your project using code like this:

```python
# import the main class
from pyhtk.HTKFeat import MFCC_HTK 

# these are additional libraries for this example
from pathlib import Path
import numpy as np

# contstruct the main mfcc object
# you can also change some arguments inside
mfcc = MFCC_HTK()

# load the raw audio file
sig = mfcc.load_raw_signal(Path("file.raw"))

# here we calculate the MFCC+energy, deltas and acceleration coefficients
feat = mfcc.get_feats(sig)
delta = mfcc.get_delta(feat, 2)
acc = mfcc.get_delta(delta, 2)

# merge the MFCCs and deltas together to get 39 features
feat = np.hstack((feat, delta, acc))

print(feat.shape)
```

## How does it work?

Open the HTKFeaturesExplained notebook in the notebooks folder and play around with it. Everythin is explained there.

Also the the `test.py` script in the tests folder compares this library to the original HTK code so you can validate its accuracy.

## Who made this?

If you have any questions, feel free to contact me at: danijel@pja.edu.pl
