'''
Mask R-CNN
Train on the cell instance segmentation dataset from kaggle 
https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

----------------------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from 
       the command line as such:

       # Train a new model starting from ImageNet weights
       python3 cell.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

       # Train a new model starting from specific weights file
       python3 cell.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

       # Resume training a model that you had trained earlier
       python3 cell.py train --dataset=/path/to/dataset --subset=train --weights=<last or path/to/weights.h5>
'''
if __name__ == 'main':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import os
#from samples.nucleus.nucleus import mask_to_rle
#from samples.balloon.balloon import COCO_WEIGHTS_PATH, DEFAULT_LOGS_DIR, ROOT_DIR
import sys
#import json
import datetime
import numpy as np
#import skimage.io
from imgaug import augmenters as iaa
import pandas as pd

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coo.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Result directory
# Save submission files here

RESULTS_DIR = os.path.join(ROOT_DIR, "results/cells/")

DATASET_DIR = os.path.join(ROOT_DIR, "cell/train.csv")
# Directory of train.csv file
#TRAIN = os.path.join(ROOT_DIR, 'train.csv')
#train = pd.read_csv(TRAIN)

# Original Image Dimension
HEIGHT = 520
WIDTH = 704
SHAPE = (HEIGHT, WIDTH)

# Target Image Dimension which are divisable by 64 as required by the MASK-RCNN model
HEIGHT_TARGET = 576
WIDTH_TARGET = 704
SHAPE_TARGET = (HEIGHT_TARGET, WIDTH_TARGET)

BATCH_SIZE = 1
#N_SAMPLES = train['id'].unique()
N_SAMPLES = 400

# Debug mode for fast experimenting with 50 samples
DEBUG = False
DEBUG_SIZE = 50

EPOCHS_ALL = 10 if DEBUG else 20

#############################################################
# One hot encoding
#############################################################
'''
Encode categorical features as a on-hot numeric array.
The encoder derives the categories based on the unique values in each feature
'''
def cell_types(train_dir):
    train = pd.read_csv(train_dir)
    train['file_path'] = train['id'].apply(get_file_path)

    # Unique cell names
    cell_names = np.sort(train['cell_type'].unique())

    # Cell type to label dictionary
    cell_name_dict = dict([(v,k) for k, v in enumerate(cell_names)])
    
    # Add cell type label to train " + 1" because label 0 is reserved for background
    train['cell_type_label'] = train['cell_type'].apply(cell_name_dict.get) + 1

    # Image ID to Cell Type Label Dictionary
    id2cell_label = dict(
        [(k, v) for k, v in train[['id', 'cell_type_label']].itertuples(name=None, index=False)]
    )

    return cell_names, id2cell_label


def get_file_path(image_id):
    return f'{ROOT_DIR}/cell/train/{image_id}.png'


##############################################################
# Configurations
##############################################################

class CellConfig(Config):
    """Configuration for training on the cell segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "cell"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6

    # Number of classes (including background)

    # number of cell types = 3

    #NUM_CLASSES = 1 + len(CELL_NAMES)  # Background + number of cell types
    NUM_CLASSES = 1 + 3

    # Number of training and validation steps per epoch
    # STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    # VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class CellInferenceConfig(CellConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

######################################################################
# Dataset
######################################################################

class CellDataset(utils.Dataset):

    def load_cell(self, dataset_dir, subset):
        
        '''
        Load a subset of the cell dataset

        dataset_dir: Root directory of the dataset "../train/"
        subset: Subset to load. 
        '''
        dataset_train_dir = os.path.join(dataset_dir, subset)

        image_paths = next(os.walk(dataset_train_dir))[2] # Get all filenames from the train directory

        train_dir = os.path.join(dataset_dir, 'train.csv')

        cell_names, id2cell_label = cell_types(train_dir)

        # Add classes. We have multiple classes
        for i, name in enumerate(cell_names):
            self.add_class("cell", 1 + i, name)

        for image in image_paths:
            self.add_image(
            "cell",
            image_id= image.split(".")[0], # we only want to have the file name as an image_id without the extension ".png"
            path = os.path.join(dataset_train_dir,image),
            width = WIDTH,
            height = HEIGHT,
            label = id2cell_label[image.split(".")[0]]
            )

  
    def load_mask(self, image_id):
        '''Generate instance mask for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        '''
        info = self.image_info[image_id]

        image_id = info['id']

        # Get masks by image_id
        masks = rle_decode(image_id, DATASET_DIR)
        masks = padding_image(masks, 0)

        # Get label

        _,_,size = masks.shape
        label = info['label']
        class_ids = np.full(size, label, dtype=np.int32)

        return masks, class_ids

#######################################################################
# Training
#######################################################################

def train(model, dataset_dir, subset):
    '''
    Train the model
    '''
    # Training dataset
    dataset_train = CellDataset()
    dataset_train.load_cell(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellDataset()
    dataset_val.load_cell(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    augmentation = iaa.SomeOf((0,2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270)
        ]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # If starting from imagenet, train heads only for a bit
    # since they have random weights

    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads'
    )

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all'
    )


#######################################################################
# RLE Encoding
#######################################################################

def padding_image(image, constant_values):
        '''
        Funtion to pad images and masks
        '''
        pad_h = (HEIGHT_TARGET - HEIGHT) // 2
        pad_w = (WIDTH_TARGET - WIDTH) // 2

        if len(image.shape) == 3:
            return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0,0)), constant_values=constant_values)

        else:
            return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=constant_values)

def rle_decode(image_id, dataset_dir):
        train = pd.read_csv(dataset_dir)

        rows = train.loc[train['id'] == image_id]

        # Image Shape
        mask = np.full(shape=[len(rows), np.prod(SHAPE)], fill_value=0, dtype=np.uint8)

        for idx, (_, row) in enumerate(rows.iterrows()):
            s = row['annotation'].split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            for lo, hi in zip(starts, ends):
                mask[idx, lo:hi] = True

        mask = mask.reshape([len(rows), *SHAPE])
        mask = np.moveaxis(mask, 0, 2)
        return mask

def mask_to_rle(source_id, mask, score):
    re = [source_id, score, mask]
    return re

#######################################################################
# Detection
#######################################################################

def detect(model, dataset_dir, subset):
    """
    Run detection on images in the given directory
    """
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = CellDataset
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()

    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['roi'], r['masks'], r['class_ids'],
            dataset.class_name, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions"
        )
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info['image_id']["id"]))

    # Save to csv file
    submission = "ImageId, EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


#######################################################################
# Command Line
#######################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for cell instance segmentation'
    )
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'"
    )
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset'
    )
    parser.add_argument('--weights', required=True,
                        metavar="path/to/weights.h5",
                        help="Path to weights.h5 file or 'coco'"
    )
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)'
    )
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help = "Subset of dataset to run prediction on"
    )
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CellConfig()
    else:
        config = CellInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs
        )
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs
        )

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from imagenet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", 
            "mrcnn_bbox", "mrcnn_mask"
        ])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print(" '{}' is not recognized. " 
              "Use 'train' or 'detect'".format(args.command))

    
