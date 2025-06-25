import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
import skimage.morphology as morpho
# from DSEC.dataset.sequence import Sequence
from joblib import Parallel, delayed
from tqdm import tqdm

from DSEC.dataset.representations import VoxelGrid
from .utils import mvsecLoadRectificationMaps, mvsecRectifyEvents, mvsecCumulateSpikesIntoFrames, events_to_voxel_grid_pytorch
from .indices import *
import datasetsmvsec.data_util as data_util


def mvsec_dataset(root: str, scenario:str, split:str, num_frames_per_depth_map, warmup_chunks, train_chunks,
               transform=None, normalize=False, learn_on='LIN', load_test_only=False):
    """
        Load a split of MVSEC in only one function.

        Sequences and indices follow those presented in Tulyakov et al. (ICCV 2019),
            "Learning an Event Sequence Embedding for Dense Event-Based Deep Stereo"
    """

    # Indices for validation and test sets vary depending on the split
    # Training and validation/test sequences also vary with the split
    if split == '1':
        training_sequences = ['2']
        valtest_sequence = '1'
        valid_indices = SPLIT1_VALID_INDICES
        test_indices = SPLIT1_TEST_INDICES
    elif split == '2':
        training_sequences = ['1', '3']
        valtest_sequence = '2'
        valid_indices = SPLIT2_VALID_INDICES
        test_indices = SPLIT2_TEST_INDICES
    elif split == '3':
        training_sequences = ['1','2']
        valtest_sequence = '3'
        valid_indices = SPLIT3_VALID_INDICES
        test_indices = SPLIT3_TEST_INDICES

    # load all indoor_flying sequences and create datasets
    # return all three train, valid, and test sets for training
    if not load_test_only:
         
                dataset1 = MVSEC_sequence(root=root,
                                        scenario=scenario, split=split, sequence=training_sequences[0],
                                        num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks, train_chunks=train_chunks,
                                        normalize=normalize, transform=transform, learn_on=learn_on, is_train = True)
        
                dataset3 = MVSEC_sequence(root=root,
                                            scenario=scenario, split=split, sequence=valtest_sequence,
                                            num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks, train_chunks=train_chunks,
                                            normalize=normalize, transform=transform, learn_on=learn_on, is_train = False)
              

            
                # dataset2 = MVSEC_sequence(root=root,
                #                           scenario=scenario, split=split, sequence=training_sequences[1],
                #                           num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks, train_chunks=train_chunks,
                #                           normalize=normalize, transform=transform, learn_on=learn_on)

                # train_set = torch.utils.data.ConcatDataset(datasets=[dataset1,dataset2])
                # # valid_set = torch.utils.data.Subset(dataset3)
                        # test_set = torch.utils.data.Subset(dataset1, test_indices)

                return dataset1, dataset3

    # only return test set for evaluation
    # else:
    #     dataset3 = MVSEC_sequence(root=root,
    #                               scenario=scenario, split=split, sequence=valtest_sequence,
    #                               num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks,
    #                               train_chunks=train_chunks,
    #                               normalize=normalize, transform=transform, learn_on=learn_on)

    #     test_set = torch.utils.data.Subset(dataset3, test_indices)

        # return test_set


class MVSEC_sequence(Dataset):
    """
    Neuromorphic dataset class to hold MVSEC data.

    Raw events are initially represented in Adress Event Representation (AER) format, that is, as a list of tuples
    (X, Y, T, P).
    An MVSEC sequence (e.g. 'indoor_flying3') is cut into frames on which we accumulate spikes occurring during a
    certain time interval dt to constitute a spike frame.
    In our paper, dt = 50 ms, which happens to correspond to the frequency of ground truth depth maps provided by the
    LIDAR.

    One chunk corresponds to a duration of 50 ms, containing 'num_frames_per_depth_map' frames for one single label.
    Essentially, the sequence is translated in 2 input tensors (left/right) of shape [# chunks, # of frames, 2 (ON/OFF), W, H].
    Corresponding ground-truth depth maps are contained in a tensor of shape [# of chunks, W, H].
    Dataloaders finally add the batch dimension.

    Warmup chunks are chronologically followed by train chunks. Warmup chunks can be used for training recurrent models;
    the idea is to deactivate automatic differentiation and perform inference on warmup chunks before train chunks, so
    that hidden states within the model reach a steady state. Then activate autodiff back before forward passing train
    chunks.

    Therefore, in our paper, we used 1 train chunk of 1 frame (of 50 ms) per depth ground truth.

    'transform' can be used for data augmentation techniques, whose methods we provide in this data_augmentation.py file
    """

    @staticmethod
    def get_wh():
        return 346, 260

    def __init__(self, root: str, scenario: str, split: str, sequence: str,
                 num_frames_per_depth_map=1, warmup_chunks=5, train_chunks=5,
                 transform=None, normalize=False, learn_on='LIN', is_train = True):

        # print("\n#####################################")
        # print("# LOADING AND PREPROCESSING DATASET #")
        # print("#####################################\n")

        self.root = root
        self.num_frames_per_depth_map = num_frames_per_depth_map

        # self.N_warmup = warmup_chunks
        # self.N_train = train_chunks

        self.transform = transform

        # load the data
        datafile = self.root + '{}/{}{}_data.hdf5'.format(scenario, scenario, sequence)
        # datafile = '/home/thc/ess/E2depth/events_rect_outday_day1.hdf5'
        data = h5py.File(datafile, 'r')
        datafile_gt = self.root + '{}/{}{}_gt.hdf5'.format(scenario, scenario, sequence)
        data_gt = h5py.File(datafile_gt, 'r')
       

        # get the ground-truth depth maps (i.e. our labels) and their timestamps
        # Ldepths_rect = np.array(data_gt['davis']['left']['depth_image_rect'])  # RECTIFIED / LEFT
        Ldepths_rect_ts = np.array(data_gt['davis']['left']['depth_image_rect_ts'])
        gray_index = np.array(data['davis']['left']['image_raw_event_inds'])

        # remove depth maps occurring during take-off and landing of the drone (bad data)
        # start_idx, end_idx = SEQUENCES_FRAMES[scenario]['split' + split][scenario + sequence]  # e.g., 'indoor_flying'/'split1'/'indoor_flying1'
        # Ldepths_rect = Ldepths_rect[2600:, :, :]
        # Ldepths_rect_ts = Ldepths_rect_ts[2600:]

        # fill holes (i.e., dead pixels) in the groundtruth with mathematical morphology's closing operation
        # yL has shape (num_labels, 1, 260, 346)
        # for i in range(len(Ldepths_rect)):
        #     filled = morpho.area_closing(Ldepths_rect[i], area_threshold=24)
        #     Ldepths_rect[i] = filled

        # pixels with zero value get a NaN value because they are invalid
        # Ldepths_rect[Ldepths_rect == np.nan] = 0.0

        # convert linear (metric) depth to log depth or disparity if required

        # shape of each depth map: (H, W) --> (1, H, W)
        # Ldepths_rect = np.expand_dims(Ldepths_rect, axis=1)

        # get the events
        # N = 100000"
        # Levents = np.array(data['davis']['left']['events'])  # EVENTS: X Y TIME POLARITY
        # # first_spike_time = Levents[0, 2]
        # # Levents[:, 2] -= first_spike_time
        # # Ldepths_rect_ts[:] -= first_spike_time
        # index = []
        
        # for numchunk in tqdm(range(len(Ldepths_rect_ts))):#12197
        #     filt_events_idx = [0,0]
        #     ts = Ldepths_rect_ts[numchunk]
        #     filt_events = Levents[ (Levents[:, 2] < ts)]
          
        #     filt_events_idx[0] = filt_events.shape[0]-3250
        #     filt_events_idx[1] = filt_events.shape[0]+3250
        #     index.append(filt_events_idx)
        # np.savetxt('/home/thc/ess/ess-main/6500index6.txt', index)
        if sequence == '2':
             event_index = np.loadtxt('/home/thc/ess/ess-main/6500index2.txt')
        else:
             event_index = np.loadtxt('/home/thc/ess/ess-main/6500index.txt')
        self.event_index = event_index
        

            
        #Revents = np.array(data['davis']['right']['events'])

        # remove events occurring during take-off and landing of the drone as well
        # Levents = Levents[(Levents[:, 2] > Ldepths_rect_ts[0])&(Levents[:, 2] < Ldepths_rect_ts[-1])]
        # Levents = Levents[(Levents[:, 2] < Ldepths_rect_ts[-1])]
        #Revents = Revents[(Revents[:, 2] > Ldepths_rect_ts[0] - 0.05) & (Revents[:, 2] < Ldepths_rect_ts[-1])]

        # rectify the spatial coordinates of spike events and get rid of events falling outside of the 346x260 fov
        Lx_path = self.root + '{}/{}_calib/{}_left_x_map.txt'.format(scenario, scenario, scenario)
        Ly_path = self.root + '{}/{}_calib/{}_left_y_map.txt'.format(scenario, scenario, scenario)
        # Rx_path = self.root + '{}/{}_calib/{}_right_x_map.txt'.format(scenario, scenario, scenario)
        # Ry_path = self.root + '{}/{}_calib/{}_right_y_map.txt'.format(scenario, scenario, scenario)
        # Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
        Lx_map, Ly_map = mvsecLoadRectificationMaps(Lx_path, Ly_path)
        # rect_Levents = np.array(mvsecRectifyEvents(Levents, Lx_map, Ly_map))



# 创建一个HDF5文件
        # with h5py.File('events_rect_outday_day1.hdf5', 'w') as f:
        #     # 创建一个名为 dataset1 的数据集，大小为 (100, )
        #     dataset1 = f.create_dataset('events', shape=(49102765+38579726,4), dtype='d')



# 打开一个已有的HDF5文件
        # with h5py.File('events_rect_outday_day1.hdf5', 'a') as f:
        #     # 获取数据集 dataset1
        #     dataset1 = f['events']
        #     # 将一个numpy数组写入数据集 dataset1
        #     dataset1[49102765:(49102765+38579726)] = rect_Levents
        #     # dataset1 = dataset1[0:(49102765+38579726)]
        
    




        # rect_Revents = np.array(mvsecRectifyEvents(Revents, Rx_map, Ry_map))






        # convert data to a sequence of frames
        # x =  rect_Levents[:,0]
        # y = rect_Levents[:,1]
        # p = rect_Levents[:,3]
        # t = rect_Levents[:,2]
        # np.savetxt('/home/thc/ess/E2depth/events_'+'outdoor_day1', rect_Levents)
        # np.savetxt('/home/thc/ess/E2depth/Ldepths_rect.txt', Ldepths_rect)
        # vg =  events_to_voxel_grid_pytorch(rect_Levents, 5, 346, 260, device)
        # xL, yL = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts, num_frames_per_depth_map=num_frames_per_depth_map)

        # normalize nonzero values in the input data to have zero mean and unit variance
        # if normalize:
        #     nonzero_mask_L = xL > 0  # LEFT
        #     mL = xL[nonzero_mask_L].mean()
        #     sL = xL[nonzero_mask_L].std()
        #     xL[nonzero_mask_L] = (xL[nonzero_mask_L] - mL) / sL

            # nonzero_mask_R = xR > 0  # RIGHT
            # mR = xR[nonzero_mask_R].mean()
            # sR = xR[nonzero_mask_R].std()
            # xR[nonzero_mask_R] = (xR[nonzero_mask_R] - mR) / sR

        # assert xL.shape == xR.shape

        # store the (N_warmup + N_train) first chunks and labels for warmup and initialization
        # self.first_data_left = xL[: 1 + 2 * (
        #             self.N_warmup + self.N_train)]  # shape: (1+(2*N_warmup+N_train), nfpdm, 2, 260, 346)
        # # self.first_data_right = xR[: 1 + 2 * (self.N_warmup + self.N_train)]
        # self.first_labels = yL[: 1 + 2 * (self.N_warmup + self.N_train)]  # shape: (1+(2*N_warmup+N_train), 1, 260, 346)

        # self.data_left = xL[self.N_warmup + self.N_train:]  # shape: (n_chunks - N_warmup, nfpdm, 2, 260, 346)
        # # self.data_right = xR[self.N_warmup + self.N_train:]
        # self.labels = yL[self.N_warmup + self.N_train:]  # shape: (n_chunks - N_warmup, 1, 260, 346)
        self.event = data
        self.data_gt = data_gt
        self.depth_ts = Ldepths_rect_ts
        self.x_map = Lx_map
        self.y_map = Ly_map
        self.voxel_grid = VoxelGrid(5, 260, 346, normalize=None)
        self.gray_index = gray_index
        self.is_train = is_train
        # self.leftdata = xL
        # self.depth = yL 
        # close hf5py file properly

        # data.close()
        # data_gt.close()


    def generate_event_tensor(self, i, event_idx, event_data, x_rect, y_rect):
        event_temp = event_data[100000*i:100000*(i+1), :]
        
        # rect_Levents = np.array(mvsecRectifyEvents(event_data, self.x_map, self.y_map))
        device = "gpu:1"
        x = x_rect[100000*i:100000*(i+1)]
        y = y_rect[100000*i:100000*(i+1)]
        event_representation = events_to_voxel_grid_pytorch(x, y, event_temp, 5, 260, 346, device)
        
        self.event_tensor[(i * 5):((i+1) * 5), :, :] = event_representation
      

  

    def rectify_events(self, x: np.ndarray, y: np.ndarray, x_map, y_map ):
        x_rect = x_map[y, x]
        y_rect = y_map[y, x]

        # convert to np.array and remove spikes falling outside of the Lidar field of view (fov)
        # rect_events = rect_events[(rect_events[:, 0] >= 0)
        #                           & (rect_events[:, 0] <= 346)
        #                           & (rect_events[:, 1] >= 0)
        #                           & (rect_events[:, 1] <= 260)]
        return x_rect, y_rect


    def __len__(self):
        return len(self.event_index)

    def __getitem__(self, index):
            N = 12000
            event_idx = self.event_index[index]
            gray_index = self.gray_index
            event_idx = int(event_idx[1])
            event_tensor = None
            image_index = np.abs(gray_index - event_idx).argmin()
            # rect_Levents = np.array(mvsecRectifyEvents(Levents, Lx_map, Ly_map))
            

            Ldepths_rect = np.array(self.data_gt['davis']['left']['depth_image_rect'][index, :, :])
            gray_gt = np.array(self.event['davis']['left']['image_raw'][image_index])
            gray_gt = gray_gt.reshape(1, gray_gt.shape[0], gray_gt.shape[1])

            
            # filled = morpho.area_closing(Ldepths_rect, area_threshold=24)
            # Ldepths_rect = filled
            Ldepths_rect = np.nan_to_num(Ldepths_rect)
            # pred_d_numpy = Ldepths_rect
            # pred_d_numpy = np.nan_to_num(pred_d_numpy)
            # pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            # pred_d_numpy = pred_d_numpy.astype(np.uint8)
            # pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
            # cv2.imwrite('mvsecgt.jpg', pred_d_color)


            event_data = np.array(self.event['davis']['left']['events'][event_idx-N*20:event_idx, :])
            # Parallel(n_jobs=8, backend="threading")(
            #         delayed(self.generate_event_tensor)(i, event_idx, event_data, x_rect, y_rect) for i in range(20))
            for i in range(20):
                event_temp = event_data[i*N:(i+1)*N, :]
                pols = event_temp[:, 3]
                t = event_temp[:, 2]
                xs = event_temp[:, 0].astype(np.int)
                ys = event_temp[:, 1].astype(np.int)
                t = (t - t[0]).astype('float32')
                t = (t/t[-1])
                x_rect = self.x_map[ys, xs]
                y_rect = self.y_map[ys, xs]
                x = x_rect.astype('float32')
                y = y_rect.astype('float32')
                pols = pols.astype('float32')
                # x,y = self.rectify_events(xs, ys, self.x_map, self.y_map)
                event_representation = self.voxel_grid.convert( torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pols),
                torch.from_numpy(t))
                event_representation = event_representation[:, :192, 1:345]
 
                # original_shape_hw = event_tensor.shape[-2:]
                # event_representation = data_util.normalize_event_tensor(event_representation)

                # target_ratio = float(192) / float(352)
                # unscaled_target_height = int(original_shape_hw[1] * target_ratio)
                # cropped_height = int(original_shape_hw[0] - unscaled_target_height)

                # # cropped_width = 0
                # event_tensor = torch.from_numpy(event_tensor[:, (cropped_height//2):-(cropped_height//2), :])
                event_representation = torch.nn.functional.interpolate(event_representation[None, :, :, :],
                                                            size=(192, 344),
                                                            mode='bilinear', align_corners=True)[0, :, :, :]


                if event_tensor is None:
                    event_tensor = event_representation
                else:
                    event_tensor = np.concatenate([event_tensor, event_representation], axis=0)
            groundtruth = Ldepths_rect[:192, 1:345]
            gray_gt = gray_gt[:, :192, 1:345]
            # self.event_tensor = self.event_tensor.astype(np.float32)
            event_tensor = event_tensor.astype(np.float32)
            
            if self.transform:
                event_tensor = self.transform(event_tensor)
                groundtruth = self.transform(groundtruth)
            
            
            if self.is_train:
                 return event_tensor, gray_gt
            else:
                 return event_tensor, groundtruth

    def show(self):
        # TODO: implement show method
        pass