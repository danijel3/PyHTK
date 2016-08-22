# we set up the path to the lib here
# you can skip this if you store the files in your project
import sys

sys.path.append('../python')

# these are the only imports you really need
from HTKFeat import MFCC_HTK
from HTK import HCopy, HTKFile

# we import some extra libs here for comparison and visualization
import numpy as np
import matplotlib

matplotlib.rcParams['backend'] = 'pdf'
import matplotlib.pyplot as P

# setting up the main class
mfcc = MFCC_HTK(filter_file='filter.csv')

# here we load the raw audio file
sig = mfcc.load_raw_signal('file.raw')

# here we calculate the MFCC+energy, deltas and acceleration coefficients
feat = mfcc.get_feats(sig)
delta = mfcc.get_delta(feat, 2)
acc = mfcc.get_delta(delta, 2)

# here we merge the MFCCs and deltas together to get 39 features
feat = np.hstack((feat, delta, acc))

# here we use HTK to calculate the same thing
# you can comment this line if you don't have HTK installed
HCopy('hcopy.conf', 'file.raw', 'file.htk')

# here we load the features generate by the command above
htk = HTKFile()
htk.load('file.htk')

# calculating the difference between features
diff = feat - htk.data

# computing and dsiplaying the maximum difference between the two methods
print("Maximum difference: {}".format(np.max(np.abs(diff))))

# displaying the difference
P.pcolormesh(diff.T)
P.savefig('diff.png')
