from datasets1.mvsec_dataset import mvsec_dataset
import numpy as np

train_set, val_set= mvsec_dataset('/media/thc/Elements/', scenario='outdoor_day', split='1',
                                          num_frames_per_depth_map=1, warmup_chunks=1, train_chunks=1,
                                          transform=None, normalize=True, learn_on='LIN')

event_index = np.loadtxt('/home/thc/ess/ess-main/6500index4.txt')
event_index = event_index + 1e+08
np.savetxt('/home/thc/ess/ess-main/6500index5.txt', event_index)