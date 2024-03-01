
import os
import sys
import pickle
import numpy as np 
from matplotlib import pyplot as plt

root = f"/scratch/lg154/sseg/GLMC/output/cifar10_resnet32/{sys.argv[1]}"

epoch = sys.argv[2]

file = os.path.join(root, 'analysis{}.pkl'.format(epoch))

with open(file, 'rb') as f: 
    nc_dt = pickle.load(f)


# Plot for w_cos
k_lst = [k for k in nc_dt if k.endswith('_cos') or k.endswith('_norm') 
         # or k=='nc1_cls'
         ]
#print(nc_dt['nc1_cls'])
fig, axes = plt.subplots(nrows=3, ncols=2)
k = 0
for key in k_lst:
    cos_matrix = nc_dt[key]  # [K, K]
    
    if key in ['w_cos', 'h_cos']:
        im = axes[int(k//2), int(k%2)].imshow(cos_matrix, cmap='RdBu')
        plt.colorbar(im, ax=axes[int(k//2), int(k%2)])
        im.set_clim(vmin=-0.9, vmax=0.9)
        axes[int(k//2), int(k%2)].set_title(key)
    else: 
        axes[int(k//2), int(k%2)].bar(np.arange(len(cos_matrix)), cos_matrix)
        axes[int(k//2), int(k%2)].set_title(key)
        
    k += 1

fig.savefig(os.path.join(root,'cos{}.png'.format(epoch)))



