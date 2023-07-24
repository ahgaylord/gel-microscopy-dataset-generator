""" Handles image and video annotations with input parameters.

@author Amory Gaylord
"""

# imports
import json
import os
import cv2
import numpy as np
import pandas as pd
import skimage.segmentation as seg
import skimage.color as color
from lib.JSON2YOLO import general_json2yolo as j2y

# define defaults
RED_LOWER = [0, 10, 5]
RED_UPPER =[255, 255, 255]
EPS = 0.0001
MIN_CLUSTER_SIZE = 20
CONTRAST = 1
BRIGHTNESS = 50
GAUSSIAN = 50
SEGMENTS = 600
SAVE_INTERMEDIATE = False
INTERMEDIATE_DESTINATION = 'vid/intermediate.png'
DRAW_DESTINATION = ''
DRAW_ON_SEGMENTS = False
DRAW_ON_ORIGINAL = False

def adjust_image(image, contrast, brightness, gaussian):
    """ Adjusts and denoises an image and returns the adjusted image.

    Parameters:
    -----------
    image : Mat
        The image to adjust.
    contrast : float
        The contrast scaling factor.
    brightness : float
        The brightness scaling factor.
    gaussian : float
        The amount to denoise.
    
    Returns:
    --------
    Mat
        The adjusted and denoised image.

    """

    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    denoised = cv2.fastNlMeansDenoisingColored(adjusted, hColor=gaussian)
    return denoised

def mask_image(image, lower, upper):
    """ Gets a binary image from an HSV image by getting only pixels in range.

    Parameters:
    -----------
    image : Mat
        The image to mask.
    lower : ndarray
        Numpy array representing the BGR color defining the lower threshold for the image.
    upper : ndarray
        Numpy array representing the BGR color defining the upper threshold for the image.

    Returns:
    --------
    mask : Mat
        The binary (1 channel) masked image where pixels in range are White (255) and pixels out of
        range are Black (0)
    
    """

    # get regions within range
    if type(lower) is np.ndarray or type(lower) is list:
        lower = np.array([int(lower[0]), int(lower[1]), int(lower[2])])
        upper = np.array([int(upper[0]), int(upper[1]), int(upper[2])])
    else:
        lower = int(lower)
        upper = int(upper)

    mask = cv2.inRange(image, lower, upper)

    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def get_clusters(image, lower_bound, upper_bound, eps, min_cluster_size, contrast, brightness, gaussian, segments, save_intermediate, intermediate_destination, draw_destination, draw_on_segment, draw_on_original, skip_slice=False, use_bw=False, return_frame=False):
    """ Gets approximate outlines for clusters in an image and returns them. Optionally returns the 
    frame instead of the bboxes and segmentation points.
    
    Parameters: 
    -----------
    image: str OR mat
        The image to get clusters from. Supports both filename input AND pre-read image input.
    lower_bound : list
        List of length 3 containing the lower threshold for detection in the BGR color space. 
    upper_bound : list
        List of length 3 containing the upper threshold for detection in the BGR color space.
    eps : float
        The accuracy modifier for polygon outlines. Defined as the maximum distance of each line
        segment outlining a cluster. Smaller values result in more curvature and vertices.
    min_cluster_size : int
        The minimum number of pixels for a cluster to qualify as a cluster.
    contrast : float
        The contrast scaling factor. 
    brightness : float
        The brightness scaling factor.
    gaussian : float
        The amount to denoise.
    segments : int
        The number of segments to break the image into.
    save_intermediate : bool
        Boolean representing whether or not to keep a saved copy of the segmented image in the
        directory.
    intermediate_destination : str
        Location to save the intermediate segmented image to.
    draw_destination : str
        Location to save the outlined clusters to as an image. If not an empty string, image will 
        be saved to this location.
    draw_on_segment : bool
        True if the image with contours drawn on it should be overlaid on the segmented intermediate
        image.
    draw_on_original : bool
        True if the image with contours drawn on it should be overlaid on the original image. This
        value will take precedence over draw_on_segment (e.g. if both are True, the overlay will be
        on the original image).
    skip_slice : bool, optional
        True if requesting to skip the segmentation step during preprocessing. False otherwise. The
        default value is False.
    use_bw : bool, optional
        True if requesting to use black and white values instead of color thresholding (in which 
        case, uses a default range of 25 to 255 grayscale). The default value is False.
    return_frame : bool, optional
        True if the program should return a Mat object with outlines instead of contour information.
        The default value is False.

    Returns:
    --------
    boxes : list
        List of bounding boxes for each contour.
    contours : list
        List of contours.
    OR
    frame : Mat
        The annotated image.

    """

    # initialize
    img = []

    # read image if necessary
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    
    # adjust the image
    adjusted = adjust_image(img, contrast, brightness, gaussian)

    if segments == 0:
        skip_slice = True

    if not skip_slice:
        # segment the image
        adjusted = seg.slic(adjusted, n_segments=segments)
        # save segmented image with avg colors
        cv2.imwrite(intermediate_destination, color.label2rgb(adjusted, img, kind='avg'))
    else:
        # skip segmentation step and write intermediate as original image
        cv2.imwrite(intermediate_destination, adjusted)
    
    # define color thresholds
    # if type(lower_bound) is list:
    #     lower_bound = np.array(lower_bound)
    # if type(upper_bound) is list:
    #     upper_bound = np.array(upper_bound)

    # read intermediate image
    sliced_color = cv2.imread(intermediate_destination)

    # Convert frame to HSV color space
    sliced_hsv = cv2.cvtColor(sliced_color, cv2.COLOR_BGR2HSV)

    if not use_bw:
        # mask image
        mask = mask_image(sliced_hsv, lower_bound, upper_bound)
    else:
        # convert to grayscale
        mask = cv2.cvtColor(sliced_color, cv2.COLOR_BGR2GRAY)
        # get regions within range
        mask = cv2.inRange(mask, np.array([25]), np.array([255]))

        # Perform morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find bounding boxes of the blobs
    boxes, _ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # set up lists
    contours = []
    bboxes = []

    # for every bounding box, approximate a polygon border and append to contours
    for box in boxes:
        # approximate polygons
        polygon = cv2.approxPolyDP(box, eps*cv2.arcLength(box, True), True)

        # calculate area to determine whether we should...
        area = cv2.contourArea(polygon)

        # ... skip this cluster if its too small
        if area < min_cluster_size:
            continue

        # add to list
        contours.append(polygon)
        bboxes.append(box)

    # remove intermediate if necessary
    if not save_intermediate:
        os.remove(intermediate_destination)

    # draw the contours if necessary
    if draw_destination != '' or return_frame:
        # draw on segmented image, draw on original image, or draw on blank
        if draw_on_original:
            frame = img
        elif draw_on_segment:
            frame = sliced_color
        else:
            frame = np.zeros_like(img)

        # draw on image
        for contour in contours:
            # draw the contour if it is large enough
            if len(contour) >= 5:
                frame = cv2.drawContours(frame.copy(), [contour.copy().astype(int)], -1, (0, 255, 255), 2)
        
        # return frame or write to draw_destination
        if return_frame:
            return frame
        else:
            # save final image
            cv2.imwrite(draw_destination, frame)
    
    # return lists if not returning frame
    return [bboxes, contours]

def generate_frames(video_path, dataset_name):
    """ Generates frames from a video to be used as a dataset.

    Parameters:
    -----------
    video_path : str
        The path of the video to read images from.
    dataset_name : str
        The name of the dataset.

    Returns:
    --------
    DataFrame
        Contains information about all of the frames

    """

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = video.read()
    frame_number = 0

    # get frame shape
    shape = frame.shape
    width = shape[0]
    height = shape[1]

    # get directory path
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    # get destination folder for images
    destination_folder = "model\\datasets\\" + dataset_name + "\\train\\images"
    # get target destination (absolute path)
    folder_loc = my_absolute_dirpath + "\\" + destination_folder

    # Get total frame count
    frame_count = (video.get(cv2.CAP_PROP_FRAME_COUNT))
    max_digits = len(str(frame_count))
    
    # set up DataFrame
    df = pd.DataFrame(columns=["file_name", "height", "width", "id"])

    # for every frame:
    while ret:
        # target filename
        name = "frame" + "{num:0{width}}".format(num=frame_number, width=max_digits) + ".jpg"
        destination = folder_loc + "\\" + name

        # save image to folder
        cv2.imwrite(destination, frame)

        # write to df
        df.loc[len(df)] = [name, height, width, frame_number]
        
        # Read the next frame
        ret, frame = video.read()
        frame_number += 1

    # Release video
    video.release()

    # return DF
    return df

def generate_labels(img, image_id, df, lower_bound, upper_bound, eps, min_cluster_size, contrast, brightness, gaussian, segments, save_intermediate, intermediate_destination, draw_destination, draw_on_segment, draw_on_original):
    """ Updates a DataFrame of annotations with new rows for each new contour in the given image.
    
    Parameters:
    -----------
    img : Mat
        Image to generate labels for.
    image_id : int
        Image number for this dataset.
    df : DataFrane
        DataFrame to add the labels for this frame to.
    lower_bound : list
        List of length 3 containing the lower threshold for detection in the BGR color space. 
    upper_bound : list
        List of length 3 containing the upper threshold for detection in the BGR color space.
    eps : float
        The accuracy modifier for polygon outlines. Defined as the maximum distance of each line
        segment outlining a cluster. Smaller values result in more curvature and vertices.
    min_cluster_size : int
        The minimum number of pixels for a cluster to qualify as a cluster.
    contrast : float
        The contrast scaling factor. 
    brightness : float
        The brightness scaling factor.
    gaussian : float
        The amount to denoise.
    segments : int
        The number of segments to break the image into.
    save_intermediate : bool
        Boolean representing whether or not to keep a saved copy of the segmented image in the
        directory.
    intermediate_destination : str
        Location to save the intermediate segmented image to.
    draw_destination : str
        Location to save the outlined clusters to as an image. If not an empty string, image will 
        be saved to this location.
    draw_on_segment : bool
        True if the image with contours drawn on it should be overlaid on the segmented intermediate
        image.
    draw_on_original : bool
        True if the image with contours drawn on it should be overlaid on the original image. This
        value will take precedence over draw_on_segment (e.g. if both are True, the overlay will be
        on the original image).

    Returns:
    --------
    DataFrame
        Updated with rows added for the outlines on this frame.

    """

    # get raw data (bounding boxes, segmentation info)
    raw_labels = get_clusters(img, lower_bound, upper_bound, eps, min_cluster_size, contrast, brightness, gaussian, segments, save_intermediate, intermediate_destination, draw_destination, draw_on_segment, draw_on_original)
    
    # split into separate lists
    contours = raw_labels[0]
    boxes = raw_labels[1]

    # target format:
    # [{'segmentation': [[155, 96, 155, 270, 351, 270, 351, 96]],
    #   'area': 34104,
    #   'iscrowd': 0,
    #   'image_id': 12,
    #   'bbox': [155, 96, 196, 174],
    #   'category_id': 7,
    #   'id': 1,
    #   'ignore': 0}

    # set up indexing
    indexer = len(df)
    n = 0

    # add contours to the df
    for contour in contours:
        # set up list
        seg = [[]]

        # add each point as x, y, x, y, etc
        for x in contour:
            seg[0].append(int(x[0][0]))
            seg[0].append(int(x[0][1]))

        # get values
        area = cv2.contourArea(contour)
        category = 1
        obj_id = indexer

        # get bbox x, y, w, h
        bbox = boxes[n]
        x, y, w, h = cv2.boundingRect(bbox)
        box = [x, y, w, h]

        # don't ignore
        ignore = 0

        # set values
        df.at[indexer, 'bbox'] = box
        df.at[indexer, 'segmentation'] = seg
        df.at[indexer, 'area'] = area
        df.at[indexer, 'iscrowd'] = 0
        df.at[indexer, 'image_id'] = image_id
        df.at[indexer, 'category_id'] = category
        df.at[indexer, 'id'] = obj_id
        df.at[indexer, 'ignore'] = ignore
        
        # increment ID val
        indexer += 1
        n += 1

    # return updated DataFrame
    return df

def dump_to_json(images, annotations, location):
    """ Dumps information on images and annotations to a .json at the given location.

    Parameters
    ----------
    images : DataFrame
        Contains information about each image file.
    annotations : DataFrame
        Contains information about each outline.
    location : str
        The location to save the .json file to (must be a valid file pathname ending in .json).
    """    

    # set up categories df
    categories = pd.DataFrame(columns=["supercategory", "id", "name"])
    categories.loc[0] = ["none", 1, "blob"]

    # convert images, annotations, and categories to dicts
    dict_images = images.to_dict(orient="records")
    dict_annotations = annotations.to_dict(orient="records")
    dict_categories = categories.to_dict(orient="records")
    data = {"images": dict_images, "type": "instances", "annotations": dict_annotations, "categories": dict_categories}

    # dump into a json
    with open(location, 'w') as outfile:
        json.dump(data, outfile)

def coco_2_yolo(folder):
    """ Converts COCO-format annotations to a YOLO-format directory of labels.

    Parameters:
    -----------
    images : DataFrame
        Contains information about all the images.
    annotations : DataFrame
        Contains information about all the outlines.
    location : str
        Absolute path to write the .json file to (likely ./model/datasets/<this_dataset>/train/labels)
    
    Returns:
    --------
    None.
    
    """

    # call conversion method
    j2y.convert_coco_json(json_dir=folder, use_segments=True)

def generate_video_labels(source, dataset_name, lower_bound, upper_bound, eps, min_cluster_size, contrast, brightness, gaussian, segments, save_intermediate, intermediate_destination, draw_destination, draw_on_segment, draw_on_original):
    """ Generates a DataFrame containing information about the labels in this video.
    
    Parameters:
    -----------
    source : str
        Source file to read video from.
    dataset_name : str
        Name of the dataset.
    lower_bound : list
        List of length 3 containing the lower threshold for detection in the BGR color space. 
    upper_bound : list
        List of length 3 containing the upper threshold for detection in the BGR color space.
    eps : float
        The accuracy modifier for polygon outlines. Defined as the maximum distance of each line
        segment outlining a cluster. Smaller values result in more curvature and vertices.
    min_cluster_size : int
        The minimum number of pixels for a cluster to qualify as a cluster.
    contrast : float
        The contrast scaling factor. 
    brightness : float
        The brightness scaling factor.
    gaussian : float
        The amount to denoise.
    segments : int
        The number of segments to break the image into.
    save_intermediate : bool
        Boolean representing whether or not to keep a saved copy of the segmented image in the
        directory.
    intermediate_destination : str
        Location to save the intermediate segmented image to.
    draw_destination : str
        Location to save the outlined clusters to as an image. If not an empty string, image will 
        be saved to this location.
    draw_on_segment : bool
        True if the image with contours drawn on it should be overlaid on the segmented intermediate
        image.
    draw_on_original : bool
        True if the image with contours drawn on it should be overlaid on the original image. This
        value will take precedence over draw_on_segment (e.g. if both are True, the overlay will be
        on the original image).
    
    Returns:
    --------
    DataFrame
        Contains segmentation points, area, iscrowd, image_id, bbox, category_id, id, and ignore.

    """

    # Target format:
    # [{'segmentation': [[155, 96, 155, 270, 351, 270, 351, 96]],
    #   'area': 34104,
    #   'iscrowd': 0,
    #   'image_id': 12,
    #   'bbox': [155, 96, 196, 174],
    #   'category_id': 7,
    #   'id': 1,
    #   'ignore': 0}

    # Open the video file
    video = cv2.VideoCapture(source)

    # set up DataFrame
    df = pd.DataFrame(columns=["segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id", "ignore"])

    # start reading video
    ret, frame = video.read()
    n = 0

    # for each frame
    while ret:
        # update labels dataframe
        df = generate_labels(frame, n, df, lower_bound, upper_bound, eps, min_cluster_size, contrast, brightness, gaussian, segments, save_intermediate, intermediate_destination, draw_destination, draw_on_segment, draw_on_original)

        # update values
        n += 1
        ret, frame = video.read()

    # release video
    video.release()

    # return list of labels
    return df

def move_files(current_loc, new_loc, percent, file_type):
    """ Moves a percent of the files of the given type in the current_loc to the new_loc.

    Parameters:
    -----------
    current_loc : str
        Absolute path to the folder containing the files to move.
    new_loc : str
        Absolute path to the folder to move the files to.
    percent : float
        Percent of the files to move to the new location. 0 < x <= 1 bounds.
    file_type : str
        File type to move; e.g. ".jpg", ".txt", ".json", etc..

    Returns:
    --------
    bool
        True if the move was successful, False if the percent was invalid. 
    
    """

    # get a float if not given as a float 0 <= x <= 1
    while percent > 1:
        percent = percent / 100

    # get list of files
    files = [f for f in os.listdir(current_loc) if file_type in f.lower()]
    total = len(files)

    # calculate number to move
    move_num = total * percent

    # can't move less than 0
    if move_num < 1:
        return False
    
    # get list of files to move
    move_list = []
    i = total - 1
    counter = 0

    while counter < move_num:
        move_list.append(files[i])
        i = i - 1
        counter += 1

    # move files
    for f in move_list:
        if os.path.exists(new_loc + f):
            continue
        else:
            os.rename(current_loc + f, new_loc + f)

    # success
    return True 

def fix_directories(percent_validation, dataset_name):
    """ Moves YOLO-format labels to the correct labels directory and splits the dataset into
    training and validation.

    Parameters:
    -----------
    percent_validation : float
        The percent of the dataset to move to the validation set. 0 < x <= 1 bounds.
    dataset_name : str
        Name of the dataset.

    Returns:
    --------
    None.
    
    """
    
    # J2Y puts labels in ./new_dir

    # get path names
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    generated_yolo_labels_path = my_absolute_dirpath + "\\new_dir\\labels\\"
    generated_yolo_imgs_path = my_absolute_dirpath + "\\new_dir\\images\\"
    new_path = my_absolute_dirpath + "\\model\\datasets\\" + dataset_name 
    training = new_path + "\\train\\"
    validation = new_path + "\\validate\\"
    
    img_train_path = training+ "images\\"
    labels_train_path = training + "labels\\"
    img_val_path = validation + "images\\"
    labels_val_path = validation + "labels\\"
    
    # move generated files to new directory
    move_files(generated_yolo_labels_path, labels_train_path, 1, ".txt")
    
    # move portion of images and labels to validation
    move_files(labels_train_path, labels_val_path, percent_validation, ".txt")
    move_files(img_train_path, img_val_path, percent_validation, ".jpg")   

    # remove extraneous directories
    os.rmdir(generated_yolo_labels_path)
    os.rmdir(generated_yolo_imgs_path)
    os.rmdir(my_absolute_dirpath + "\\new_dir\\")

def set_up_directories(dataset_name):
    """ Creates directories relative to this file for training and validation.

    Parameters:
    -----------
    dataset_name : str
        Name of this dataset
    
    Returns:
    --------
    str
        The absolute path to the dataset folder.
    
    """

    # get directory paths
    abs_path = os.path.abspath(os.path.dirname(__file__))
    vid_folder = abs_path + "\\vid\\"
    model_folder = abs_path + "\\model\\"
    datasets_folder = model_folder + "datasets\\"
    dataset_folder = abs_path + "\\model\\datasets\\" + dataset_name 
    training = dataset_folder + "\\train\\"
    validation = dataset_folder + "\\validate\\"
    
    img_train_path = training+ "images\\"
    labels_train_path = training + "labels\\"
    img_val_path = validation + "images\\"
    labels_val_path = validation + "labels\\"

    # create directories
    if not os.path.exists(vid_folder):
        os.mkdir(vid_folder)
    
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    
    if not os.path.exists(datasets_folder):
        os.mkdir(datasets_folder)
        
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    if not os.path.exists(training):
        os.mkdir(training)
    
    if not os.path.exists(validation):
        os.mkdir(validation)

    if not os.path.exists(img_train_path):
        os.mkdir(img_train_path)

    if not os.path.exists(img_val_path):
        os.mkdir(img_val_path)
    
    if not os.path.exists(labels_train_path):
        os.mkdir(labels_train_path)
                
    if not os.path.exists(labels_val_path):
        os.mkdir(labels_val_path)

    return dataset_folder

def generate_video(source, save_loc, lower_bound=RED_LOWER, upper_bound=RED_UPPER, eps=EPS, min_cluster_size=MIN_CLUSTER_SIZE, contrast=CONTRAST, brightness=BRIGHTNESS, gaussian=GAUSSIAN, segments=SEGMENTS):
    """ Generates an annotated video and saves it to the save location.

    Parameters
    ----------
    source : str
        Path to the source file. May be absolute or relative.
    save_loc : str
        Path to the location to save the annotated video.
    lower_bound : list, optional
        Lower color threshold for detection. The default value is RED_LOWER.
    upper_bound : _type_, optional
        Upper color threshold for detection. The default value is RED_UPPER.
    eps : float, optional
        Used to define the minimum distance between two identified outline points (e.g. the
        vertices of the polygon outlining a cluster). The default value is EPS.
    min_cluster_size : int, optional
        The minimum number of pixels for a cluster to be counted as a cluster. The default value is 
        MIN_CLUSTER_SIZE.
    contrast : int, optional
        Amount to increase contrast by in preprocessing. The default value is CONTRAST.
    brightness : int, optional
        Amount to increase brightness by in preprocessing. The default value is BRIGHTNESS.
    gaussian : int. The optional
        Amount to blur the image in preprocessing. The default value is GAUSSIAN.
    segments : int, optional
        The number of segments to split the image into during preprocessing. If 0, the segmentation
        preprocessing step is skipped. The default value is SEGMENTS.

    """

    # open video capture object
    vid = cv2.VideoCapture(source)

    # read first frame
    ret, frame = vid.read()

    # get video properties
    fps = vid.get(cv2.CAP_PROP_FPS)

    shape = frame.shape
    width = shape[0]
    height = shape[1]
    frame_size = (width, height)

    # determine whether to skip segmentation in preprocessing step
    skip_slic = False

    if segments == 0:
        skip_slic = True

    # create videowriter object
    out = cv2.VideoWriter(save_loc, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, frame_size)
    
    # for each frame
    while ret:
        # get clusters and return frame
        f = get_clusters(frame, lower_bound, upper_bound, eps, min_cluster_size, contrast, brightness, gaussian, segments, False, INTERMEDIATE_DESTINATION, DRAW_DESTINATION, False, True, return_frame=True, skip_slice=skip_slic)

        # write to video
        out.write(f)

        # update frame and ret
        ret, frame = vid.read()
    
    # release objects
    vid.release()
    out.release()

def generate_dataset(source, dataset_name, json_name, lower_bound=RED_LOWER, upper_bound=RED_UPPER, eps=EPS, min_cluster_size=MIN_CLUSTER_SIZE, contrast=CONTRAST, brightness=BRIGHTNESS, gaussian=GAUSSIAN, segments=SEGMENTS, save_intermediate=SAVE_INTERMEDIATE, intermediate_destination=INTERMEDIATE_DESTINATION, draw_destination=DRAW_DESTINATION, draw_on_segment=DRAW_ON_SEGMENTS, draw_on_original=DRAW_ON_ORIGINAL):
    """

    Parameters:
    -----------
    source : str
        Path to the source video to segment.
    dataset_name : str
        Title to use to name the dataset.
    json_name : str
        What to name the .json file (e.g. "labels.json"). Do not include directory information.
    lower_bound : list, optional
        List of length 3 containing the lower threshold for detection in the BGR color space. 
        The default value is [0, 10, 5].
    upper_bound : list, optional
        List of length 3 containing the upper threshold for detection in the BGR color space. 
        The default value is [255, 255, 255].
    eps : float, optional
        The accuracy modifier for polygon outlines. Defined as the maximum distance of each line
        segment outlining a cluster. Smaller values result in more curvature and vertices. The 
        default value is 0.0001.
    min_cluster_size : int, optional
        The minimum number of pixels for a cluster to qualify as a cluster.
    contrast : float, optional
        The contrast scaling factor. The default value is 1.
    brightness : float, optional
        The brightness scaling factor. The default value is 50.
    gaussian : float, optional
        The amount to denoise. The default value is 50.
    segments : int, optional
        The number of segments to break the image into. The default value is 600.
    save_intermediate : bool, optional
        Boolean representing whether or not to keep a saved copy of the segmented image in the
        directory. The default value is False.
    intermediate_destination : str, optional
        Location to save the intermediate segmented image to. The default value is 
        '../vid/intermediate.png'.
    draw_destination : str, optional
        Location to save the outlined clusters to as an image. If not an empty string, image will 
        be saved to this location. The default value is ''.
    draw_on_segment : bool, optional
        True if the image with contours drawn on it should be overlaid on the segmented intermediate
        image. The default value is False.
    draw_on_original : bool, optional
        True if the image with contours drawn on it should be overlaid on the original image. This
        value will take precedence over draw_on_segment (e.g. if both are True, the overlay will be
        on the original image). The default value is False
    
    """

    # get directory information
    folder_loc = set_up_directories(dataset_name)
    folder_loc_train = folder_loc + "\\train"
    folder = folder_loc_train + "\\labels\\"
    json_loc = folder + json_name

    if ".avi" in source:
        # get frames from video
        images = generate_frames(source, dataset_name)

        # get labels from video
        annotations = generate_video_labels(source, dataset_name, lower_bound=lower_bound, upper_bound=upper_bound, eps=eps, min_cluster_size=min_cluster_size, contrast=contrast, brightness=brightness, gaussian=gaussian, segments=segments, save_intermediate=save_intermediate, intermediate_destination=intermediate_destination, draw_destination=draw_destination, draw_on_segment=draw_on_segment, draw_on_original=draw_on_original)
    else:
        img = cv2.imread(source)
        # set up annotations DataFrame
        annotations = pd.DataFrame(columns=["segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id", "ignore"])

        # get cluster labels
        annotations = generate_labels(img, 0, annotations, lower_bound, upper_bound, eps, min_cluster_size, contrast, brightness, gaussian, segments, False, intermediate_destination, draw_destination, draw_on_segment, draw_on_original)

        # set up images DataFrame
        images = pd.DataFrame(columns=["file_name", "height", "width", "id"])

        # get frame shape
        shape = img.shape
        width = shape[0]
        height = shape[1]

        # get directory path
        my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
        # get destination folder for images
        destination_folder = "model\\datasets\\" + dataset_name + "\\train\\images"
        # get target destination (absolute path)
        folder_loc = my_absolute_dirpath + "\\" + destination_folder

        # image name
        name = "frame.jpg"
        destination = folder_loc + "\\" + name

        # save image to folder
        cv2.imwrite(destination, img)

        # write image info to df
        images.loc[len(images)] = [name, height, width, 0]

    # save as json file
    dump_to_json(images, annotations, json_loc)
