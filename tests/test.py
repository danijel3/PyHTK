# these are the only imports you really need
from pyhtk.HTKFeat import MFCC_HTK
from tests.HTK import HCopy, HTKFile

# we import some extra libs here for comparison and visualization
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.rcParams["backend"] = "pdf"
import matplotlib.pyplot as P

# setting up the main class
mfcc = MFCC_HTK(filter_file=Path("tests/filter.csv"))

# here we load the raw audio file
sig = mfcc.load_raw_signal(Path("tests/file.raw"))

# here we calculate the MFCC+energy, deltas and acceleration coefficients
feat = mfcc.get_feats(sig)
delta = mfcc.get_delta(feat, 2)
acc = mfcc.get_delta(delta, 2)

# here we merge the MFCCs and deltas together to get 39 features
feat = np.hstack((feat, delta, acc))

# here we use HTK to calculate the same thing
try:
    HCopy("hcopy.conf", "tests/file.raw", "file.htk")
except FileNotFoundError:
    # you need HTK installed for the above to work
    print("HTK not found! Skipping...")
    # if you don't have HTK, the file is stored on the repo anyway

# here we load the features generate by the command above
htk = HTKFile()
htk.load("tests/file.htk")

# calculating the difference between features
diff = feat - htk.data

# computing and dsiplaying the maximum difference between the two methods
print("Maximum difference: {}".format(np.max(np.abs(diff))))

# displaying the difference
P.pcolormesh(diff.T)
P.savefig("tests/diff.png")
