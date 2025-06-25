import argparse

from sklearn.tree import export_graphviz

from config.settings import Settings
from training.ess_trainer import ESSModel
from models.style_networks import StyleEncoderE2VID, SemSegE2VID
from e2vid.model.model import E2VIDRecurrent
from e2vid.utils.loading_utils import load_model
from datasets.wrapper_dataloader import WrapperDataset
from torchvision import transforms
from utils.saver import CheckpointSaver
from e2vid.image_reconstructor import ImageReconstructor
from tqdm import tqdm
import torchvision

import numpy as np
import torch
import random
import os
import logging1
import metrics
import cv2


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']



class ToTensor:
    def __call__(self, data):
        for i in range(len(data)):
            data[i] = torch.from_numpy(data[i])
        # init_pots = torch.from_numpy(init_pots)
        # warmup_chunks_left = torch.from_numpy(warmup_chunks_left)
        # train_chunks_left = torch.from_numpy(train_chunks_left)
        # groundtruth = torch.from_numpy(groundtruth)

        # if type(warmup_chunks_right) != int and type(train_chunks_right) != int:
        #     warmup_chunks_right = torch.from_numpy(warmup_chunks_right)
        #     train_chunks_right = torch.from_numpy(train_chunks_right)
        return data

tsfm = transforms.Compose([
    ToTensor(),
    # RandomHorizontalFlip(p=0.5),
    # RandomVerticalFlip(p=0.5),
    # RandomTimeMirror(p=0.5),
    # RandomEventDrop(p=0.5, min_drop_rate=0., max_drop_rate=0.4)
])

def getDataloader(dataset_name):
        """Returns the dataset loader specified in the settings file"""
        if dataset_name == 'DSEC_events':
            from datasets.DSEC_events_loader import DSECEvents
            return DSECEvents
        elif dataset_name == 'Cityscapes_gray':
            from datasets.cityscapes_loader import CityscapesGray
            return CityscapesGray
        elif dataset_name == 'DDD17_events':
            from datasets.ddd17_events_loader import DDD17Events
            return DDD17Events
        elif dataset_name == 'MVSEC_events':
            from datasets1.mvsec_dataset import mvsec_dataset
            return mvsec_dataset
        elif dataset_name == 'kitti':
            from dataset.kitti import kitti
            return kitti
            # from datasetskitti import KITTIDepthDataset
            # return KITTIDepthDataset


def createkittidataset(dataset_name, dataset_path):
            databuilder = getDataloader(dataset_name)
            val_dataset = databuilder(dataset_path, filenames_path='/home/thc/ess/ess-main/', 
                 is_train=False, dataset='kitti', crop_size=(192, 344),
                 scale_size=(664, 192))            
            dataset_loader = torch.utils.data.DataLoader

            val_loader_sensor = dataset_loader(val_dataset, batch_size=1,
                                            num_workers= 4,
                                            pin_memory=False, shuffle=False, drop_last=True)
            print('kitti num of batches: ', len(val_loader_sensor))

            return  val_loader_sensor
    
def createmvsecDataset(dataset_name, dataset_path, img_size, batch_size, nr_events_window, event_representation,
                      nr_temporal_bins):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = getDataloader(dataset_name)

        train_set, val_set= dataset_builder('/media/thc/Elements/mvsec/', scenario='outdoor_day', split='1',
                                          num_frames_per_depth_map=1, warmup_chunks=1, train_chunks=1,
                                          transform=tsfm, normalize=True, learn_on='LIN')
        
        val_loader_sensor_b = torch.utils.data.DataLoader(dataset=val_set,
                                                batch_size=1,
                                                shuffle=False,
                                                drop_last=True,
                                                pin_memory=True)
        print('MVSEC num of batches: ', len(val_loader_sensor_b))

        return val_loader_sensor_b


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    device = torch.device('cuda')
    
    result_path = '/home/thc/ess/ess-main/result0724/b/'
    logging1.check_and_make_dirs(result_path)
    print("Saving result images in to %s" % result_path)

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    
    print("\n1. Define Model")
    front_end_sensor_a = StyleEncoderE2VID(1,
                                                    skip_connect=settings.skip_connect_encoder)

        # Front End Sensor B
    front_end_sensor_b, e2vid_decoder = load_model(settings.path_to_model)
    # modeltype = {'num_bins':5,
    #                                     'skip_type':'sum',
    #                                     'recurrent_block_type':'convlstm',
    #                                     'num_encoders':3,
    #                                     'base_num_channels':32,
    #                                     'num_residual_blocks':2,
    #                                     'use_upsample_conv':False,
    #                                     'norm':'BN'}
    # front_end_sensor_b = E2VIDRecurrent(modeltype)
    # front_end_sensor_b.load_state_dict(torch.load("/home/thc/ess/e2vid/model_params45.pth"))


    reconstructor = ImageReconstructor(front_end_sensor_b, 192, 344,
                                                settings.nr_temporal_bins_b, settings.gpu_device,
                                                settings.e2vid_config)
    task_backend = SemSegE2VID(input_c=256, output_c=settings.semseg_num_classes,
                                        skip_connect=settings.skip_connect_task,
                                        skip_type=settings.skip_connect_task_type) 
    models_dict = {"front_sensor_a": front_end_sensor_a,
                   "front_sensor_b": front_end_sensor_b,
                   "back_end": task_backend}
    optimizers_dict = {}
    saver = CheckpointSaver(save_dir=settings.ckpt_dir)
    saver.load_checkpoint(models_dict, optimizers_dict, checkpoint_file='/home/thc/ess/ess-main/<path>/20230530-140206/checkpoints/Epoch_20.pt', load_optimizer=False)

        
    out = createkittidataset(settings.dataset_name_a, settings.dataset_path_a)
    val_loader_sensor_a = out

    out = createmvsecDataset(settings.dataset_name_b,
                                settings.dataset_path_b,
                                settings.img_size_b,
                                settings.batch_size_b,
                                settings.nr_events_window_b,
                                settings.event_representation_b,
                                settings.input_channels_b )
    val_loader_sensor_b = out

    val_loader = WrapperDataset(val_loader_sensor_a, val_loader_sensor_b, device)

    print("\n3. Inference & Evaluate")
    val_loader.createIterators()
    for model in models_dict:
            models_dict[model].eval()
            models_dict[model].to(device)
    for batch_idx, batch in tqdm(enumerate(val_loader_sensor_b)):
        input, depth_gt = batch
        input = input.to(device)
        depth_gt = depth_gt.to(device)
        # depth_gt = depth_gt / 255.0
        with torch.no_grad():
            reconstructor.last_states_for_each_channel = {'grayscale': None}
            for i in range(settings.nr_events_data_b):
                event_tensor = input[:, i * settings.input_channels_b:(i + 1) * settings.input_channels_b, :,:]
                img_fake, _, content_first_sensor = reconstructor.update_reconstruction(event_tensor)

            # torchvision.utils.save_image(depth_gt,'/home/thc/ess/ess-main/result0724/c/'+ str(batch_idx) + '.png', padding=0)
            # torchvision.utils.save_image(img_fake,'/home/thc/ess/ess-main/result0724/d/'+ str(batch_idx) + '.png', padding=0)

        # content_second_sensor = models_dict['front_sensor_a'](input)
        #可以先转换图片再转换成深度验证指标
        pred = models_dict['back_end'](content_first_sensor)
        pred_d = pred[1]  * 60.0
        # depth_gt = batch[1][1]
        pred_d, depth_gt = pred_d.squeeze(), depth_gt.squeeze()
        pred_crop, gt_crop = metrics.cropping_img(pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
            
        
        save_path = os.path.join(result_path, str(batch_idx) + '.jpg')
        pred_d_numpy = pred_d.squeeze().detach().cpu().numpy()
        pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        cv2.imwrite(save_path, pred_d_color)
        logging1.progress_bar(batch_idx, len(val_loader), 1, 1)

   
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)
    display_result = logging1.display_result(result_metrics)
   
    # print("\nCrop Method: ", args.kitti_crop)
    print(display_result)

    print("Done")


if __name__ == "__main__":
    main()
