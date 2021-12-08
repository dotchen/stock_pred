import tqdm
import numpy as np

np.random.seed(42)

CLASS_FREQ = [3228,3167,2827]
CLASS_IDX = np.cumsum(CLASS_FREQ)


acc = []
for i in range(sum(CLASS_FREQ)):
	gt = np.searchsorted(CLASS_IDX, i, side='right')
	acc.append(np.random.randint(len(CLASS_FREQ))==gt)

print (np.mean(acc))
# print (sum(acc==0))
# print (sum(acc==1))
# print (sum(acc==2))

	# acc.append()
