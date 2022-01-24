from sklearn.model_selection import train_test_split

import logging, sys, os, argparse
import matplotlib.pyplot as plt
import time
from DeepEndoIBM.utils.tools import TrainDataSegReader, TestDataSegReader, display_mask
import numpy as np
from DeepEndoIBM.utils.metrics import MyMeanIOU,DiceCoeff, dice_coef

from DeepEndoIBM.config.config import SegmentationConfig, DataConfig
from DeepEndoIBM.models.get_model import GetSegmentationModel
from pathlib import Path

import yaml
import dacite
import tensorflow as tf
import PIL
import os
import pandas as pd
from PIL import ImageOps

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, random_shift,random_shear
import PIL
import dacite 
from datetime import date


def run_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "-c",
        "--config_file",
        help="path to config",
        type=str,
        default=None,

    )
    
    args = parser.parse_args()

    return args


    
def main():
    args = run_argparse()
    with open(args.config_file, "r") as file:
        pipeline_config_yaml = yaml.load(file, yaml.FullLoader)
    pipeline_config = dacite.from_dict(SegmentationConfig, pipeline_config_yaml)
    data_config = dacite.from_dict(DataConfig, pipeline_config_yaml)
    target_size= (pipeline_config.target_size, pipeline_config.target_size)
    data_name = data_config.train_path.split("/")[-1][:-4]

    
    train_set = pd.read_csv(data_config.train_path)[["abs_path_images","abs_path_masks"]]
    validation_set = pd.read_csv(data_config.val_path)[["abs_path_images","abs_path_masks"]]
    
    test_set = pd.read_csv(data_config.test_path)[["abs_path_images","abs_path_masks"]]
 
    
    
    test_generator = TrainDataSegReader(test_set,  target_size, pipeline_config.batch_size, validation=True, augmentations_config=pipeline_config.augmentations)     
  

  

    STEP_SIZE_TEST=len(test_set)//pipeline_config.batch_size
 

    my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor= pipeline_config.monitor_value, patience=pipeline_config.train_patience, restore_best_weights=True),
  ]
    
    my_metrics=[MyMeanIOU(num_classes=2),dice_coef]



    if pipeline_config.cross_validation:
        all_mean_iou_train = []
        all_mean_iou_val = []

        all_dice_coeff_train = []
        all_dice_coeff_val = []
        
        
        all_train_loss = []
        all_val_loss = []
        
        
        
        cv_dice_coeff_val = []
        cv_mean_iou_val = []
        
        
        new_train_set = pd.concat([train_set,validation_set],axis=0)
        num_val_samples = len(new_train_set) // pipeline_config.cv_split

        model_dir = data_config.model_path +"/" + data_name + "/" + "cv_" + pipeline_config.model_name + "_" + str(date.today().strftime("%d_%m_%Y"))


        start_time = time.time()
        for i in range(pipeline_config.cv_split):
            print('processing fold #', i)
            validation_set = new_train_set.iloc[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_set = pd.concat(
            [new_train_set.iloc[:i * num_val_samples],
            new_train_set.iloc[(i + 1) * num_val_samples:]], axis=0)
            STEP_SIZE_TRAIN_PARTIAL=len(partial_train_set)//pipeline_config.batch_size         
            STEP_SIZE_VALID=len(validation_set)//pipeline_config.batch_size
            
            model = GetSegmentationModel(pipeline_config)
            model.compile(optimizer=pipeline_config.optimizer, loss=pipeline_config.loss_function, metrics=my_metrics)    
            
            
            train_generator = TrainDataSegReader(partial_train_set, target_size, pipeline_config.batch_size, augmentations_config=pipeline_config.augmentations)        
            validation_generator = TrainDataSegReader(validation_set,  target_size, pipeline_config.batch_size, validation=True, augmentations_config=pipeline_config.augmentations)     

            
            
            history = model.fit(train_generator, epochs=pipeline_config.epochs,  steps_per_epoch=STEP_SIZE_TRAIN_PARTIAL,validation_data=validation_generator,validation_steps=STEP_SIZE_VALID, callbacks=my_callbacks)
            
            cv_res = model.evaluate(validation_generator, batch_size=pipeline_config.batch_size, steps=STEP_SIZE_VALID, return_dict=True)
            
            cv_dice_coeff_val.append(cv_res["dice_coef"])
            cv_mean_iou_val.append(cv_res["MyMeanIOU"])
            
            
            all_dice_coeff_train.append(history.history['dice_coef'])
            all_dice_coeff_val.append(history.history['val_dice_coef'])

            all_mean_iou_train.append(history.history["MyMeanIOU"])
            all_mean_iou_val.append(history.history["val_MyMeanIOU"])
            
            all_train_loss.append(history.history["loss"])
            all_val_loss.append( history.history["val_loss"])
            
        dice_coeff_train = [np.mean([x[i] for x in all_dice_coeff_train]) for i in range(len(all_dice_coeff_train))]
        dice_coeff_val = [np.mean([x[i] for x in all_dice_coeff_val]) for i in range(len(all_dice_coeff_train))]

        mean_iou_train = [np.mean([x[i] for x in all_mean_iou_train]) for i in range(len(all_mean_iou_train))]
        mean_iou_val = [np.mean([x[i] for x in all_mean_iou_val]) for i in range(len(all_mean_iou_val))]

        train_loss = [np.mean([x[i] for x in all_train_loss]) for i in range(len(all_train_loss))]
        val_loss = [np.mean([x[i] for x in all_val_loss]) for i in range(len(all_val_loss))]


        
        STEP_SIZE_TRAIN=len(new_train_set)//pipeline_config.batch_size         
        train_generator = TrainDataSegReader(new_train_set, target_size, pipeline_config.batch_size, augmentations_config=pipeline_config.augmentations)        
        model = GetSegmentationModel(pipeline_config)
        model.compile(optimizer=pipeline_config.optimizer, loss=pipeline_config.loss_function, metrics=my_metrics)  
        history = model.fit(train_generator, epochs=pipeline_config.epochs,  steps_per_epoch=STEP_SIZE_TRAIN,validation_data=validation_generator,validation_steps=STEP_SIZE_VALID, callbacks=my_callbacks)
        training_time = time.time() - start_time
        
        cv_mean_mean_iou_val = np.mean(cv_mean_iou_val)
        cv_mean_dice_coeff_val = np.mean(cv_dice_coeff_val)
        
    else:

        model_dir = data_config.model_path +"/" + data_name + "/" + pipeline_config.model_name + "_" + str(date.today().strftime("%d_%m_%Y"))


        train_generator = TrainDataSegReader(train_set, target_size, pipeline_config.batch_size, augmentations_config=pipeline_config.augmentations)        
        validation_generator = TrainDataSegReader(validation_set,  target_size, pipeline_config.batch_size, validation=True, augmentations_config=pipeline_config.augmentations)     
    
        STEP_SIZE_TRAIN=len(train_set)//pipeline_config.batch_size         
        STEP_SIZE_VALID=len(validation_set)//pipeline_config.batch_size
    
    
        model = GetSegmentationModel(pipeline_config)
        
    
    
        model.compile(optimizer=pipeline_config.optimizer, loss=pipeline_config.loss_function, metrics=my_metrics)    
        
        start_time = time.time()
    
        history = model.fit(train_generator, epochs=pipeline_config.epochs,  steps_per_epoch=STEP_SIZE_TRAIN,validation_data=validation_generator,validation_steps=STEP_SIZE_VALID, callbacks=my_callbacks)

        training_time = time.time() - start_time
        
        
        dice_coeff_train = history.history['dice_coef']
        dice_coeff_val = history.history['val_dice_coef']

        mean_iou_train = history.history['MyMeanIOU']
        mean_iou_val = history.history['val_MyMeanIOU']
        
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        

    if data_config.save_model:
        model.save(model_dir)
    
    results_dir = model_dir + "/results" 

    Path(results_dir).mkdir(parents=True, exist_ok=True)    
    x_axis = [i for i in range(1, len(dice_coeff_train)+1)]



    if data_config.generate_metrics:
        results = {}
        results["epochs"] = len(x_axis)
        if pipeline_config.cross_validation:
            results["val_MeanIOU"] = float(cv_mean_mean_iou_val)
            results["val_dice_coef"] = float(cv_mean_dice_coeff_val)

        
        res = model.evaluate(test_generator, batch_size=pipeline_config.batch_size, steps=STEP_SIZE_TEST, return_dict=True)    
        results["test_MeanIOU"] = res["MyMeanIOU"]
        results["test_dice_coef"] = res["dice_coef"]
 
    
        
        results["training_time_seconds"] = training_time
        with open(results_dir + "/results.yaml", 'w') as f:
            yaml.dump(results, f) 
        print(results)
    


    if data_config.generate_plots:
        plt.plot(x_axis, dice_coeff_train)
        plt.plot(x_axis, dice_coeff_val)
        plt.ylabel('dice coefficient')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(results_dir + '/dice_coefficient.png')
        plt.clf()
    
        plt.plot(x_axis, mean_iou_train)
        plt.plot(x_axis, mean_iou_val)
        plt.ylabel('mean IOU')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(results_dir + '/mean_iou.png')
        plt.clf()
    
        
        plt.plot(x_axis, train_loss)
        plt.plot(x_axis, val_loss)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(results_dir + '/loss.png')
        plt.clf()

    
    if data_config.generate_samples:
        test_path_masks = results_dir + "/generated_test_masks"
        
        
        Path(test_path_masks).mkdir(parents=True, exist_ok=True)    

        
        display_mask(model, test_set.iloc[::-1]["abs_path_images"], test_set.iloc[::-1]["abs_path_masks"], n_samples = data_config.n_samples, target_size = pipeline_config.target_size, path_to_save=test_path_masks)





if __name__ == "__main__":
    main()
