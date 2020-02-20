import numpy as np


s = [3.98,3.58,3.63,3.91,3.73,3.85,3.97,3.81,3.91,3.88,3.95,3.63,3.85,3.77,3.88,3.81,4,3.83,3.58,3.81,3.39,3.85,3.73,3.99,3.52,3.93,4,3.91]
c = [2,3,2,2,3,3,3,3,3,1,2,3,5,3,3,3,3,2,3,4,5,3,4,4,3,4,4,3]

s = np.array(s).astype(np.float32)
c = np.array(c).astype(np.float32)
ss = 0.
ts = 0.
for i in range(s.shape[0]):
    ss += s[i] * c[i]
    ts += c[i]

print(ss.sum() / ts.sum())

import numpy as np

