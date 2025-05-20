import os
import cv2
import time
import numpy as np
import math
import json
import datetime


def euc_dist(arr1, point):
    square_sub = (arr1 - point) ** 2
    return np.sqrt(np.sum(square_sub, axis=1))



def preprocess_image(img_path, variables):
    img = cv2.imread(img_path)
    img_ht, img_wd = img.shape[0], img.shape[1]
    img = cv2.resize(img, (variables.resize_wd, variables.resize_ht))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # color histogram equilization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cl1 = clahe.apply(img_gray)

    # gaussian adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    # adding all the computed required items in variables object
    variables.img_ht = img_ht
    variables.img_wd = img_wd
    variables.img_gray = img_gray
    variables.img_thresh = img_thresh
    variables.img = img
    return variables


def get_extreme_coordinates(mask):
    indices = np.where(mask == 255)
    # Extract the x and y coordinates of the pixels.
    x = indices[1]
    y = indices[0]

    # Find the minimum and maximum x and y coordinates.
    topleft = (np.min(x), np.min(y))
    bottomright = (np.max(x), np.max(y))

    return topleft, bottomright

def load_hand_poses(hand_pose_paths, hand_mask_pose_paths, variables):
    variables.hands_data = []
    if not hand_pose_paths or not hand_mask_pose_paths or len(hand_pose_paths) != len(hand_mask_pose_paths):
        print("Warning: Hand pose paths or mask paths are empty or mismatched. Hand animation might be skipped.")
        return variables

    for hand_path, hand_mask_path in zip(hand_pose_paths, hand_mask_pose_paths):
        try:
            hand_img = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED) # Load with alpha if available
            hand_mask_img = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)

            if hand_img is None:
                print(f"Warning: Could not read hand image at {hand_path}. Skipping this pose.")
                continue
            if hand_mask_img is None:
                print(f"Warning: Could not read hand mask at {hand_mask_path}. Skipping this pose.")
                continue
            
            # Ensure hand_img has 3 channels (BGR) for consistent processing
            if hand_img.shape[2] == 4: # Check for alpha channel
                # If alpha exists, one might use it or convert to BGR
                # For now, let's assume we want BGR and ignore alpha for hand itself, mask handles transparency
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGRA2BGR)
            
            # Preprocessing similar to original preprocess_hand_image
            top_left, bottom_right = get_extreme_coordinates(hand_mask_img)
            hand_img = hand_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            hand_mask_img = hand_mask_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            hand_mask_inv_img = 255 - hand_mask_img

            # Normalize masks
            hand_mask_normalized = hand_mask_img / 255.0
            hand_mask_inv_normalized = hand_mask_inv_img / 255.0

            # Make hand background black based on its mask
            # This step is crucial if the hand image itself isn't pre-masked
            hand_img_bg_black = hand_img.copy() # Create a copy to modify
            hand_img_bg_black[hand_mask_img == 0] = [0, 0, 0]


            hand_ht, hand_wd = hand_img_bg_black.shape[0], hand_img_bg_black.shape[1]

            variables.hands_data.append({
                "image": hand_img_bg_black, # Use the one with black background
                "mask_inv": hand_mask_inv_normalized, # For applying to drawing
                "height": hand_ht,
                "width": hand_wd
            })
        except Exception as e:
            print(f"Error processing hand pose {hand_path} or {hand_mask_path}: {e}")
            
    if not variables.hands_data:
        print("Warning: No hand poses were successfully loaded. Hand animation will be skipped.")
    else:
        print(f"Successfully loaded {len(variables.hands_data)} hand poses.")
        
    return variables


def draw_hand_on_img(
    drawing,
    current_hand_data, # Expects a dict from variables.hands_data
    drawing_coord_x,
    drawing_coord_y,
    img_ht, # Overall canvas height
    img_wd  # Overall canvas width
):
    hand_image = current_hand_data["image"]
    hand_mask_inv = current_hand_data["mask_inv"]
    hand_ht = current_hand_data["height"]
    hand_wd = current_hand_data["width"]

    remaining_ht = img_ht - drawing_coord_y
    remaining_wd = img_wd - drawing_coord_x
    
    crop_hand_ht = min(hand_ht, remaining_ht)
    crop_hand_wd = min(hand_wd, remaining_wd)

    if crop_hand_ht <= 0 or crop_hand_wd <= 0: # Avoid errors with zero or negative crop dimensions
        return drawing

    hand_cropped = hand_image[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]
    
    # Ensure mask is 2D for broadcasting with 3-channel drawing ROI
    if hand_mask_inv_cropped.ndim == 2:
        hand_mask_inv_cropped_rgb = cv2.cvtColor(hand_mask_inv_cropped, cv2.COLOR_GRAY2BGR)
    elif hand_mask_inv_cropped.ndim == 3 and hand_mask_inv_cropped.shape[2] == 1: # (h, w, 1)
        hand_mask_inv_cropped_rgb = cv2.cvtColor(hand_mask_inv_cropped, cv2.COLOR_GRAY2BGR)
    else: # Already (h, w, 3)
        hand_mask_inv_cropped_rgb = hand_mask_inv_cropped


    roi = drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ]

    # Apply inverse mask to the region of interest (ROI)
    # Element-wise multiplication
    masked_roi = roi * hand_mask_inv_cropped_rgb

    # Add the cropped hand image to the masked ROI
    # Ensure hand_cropped is BGR
    if hand_cropped.shape[2] == 4: # BGRA
        hand_cropped = cv2.cvtColor(hand_cropped, cv2.COLOR_BGRA2BGR)

    combined_roi = masked_roi + hand_cropped
    
    drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ] = combined_roi
    
    return drawing


def draw_masked_object(
    variables, object_mask=None, skip_rate=5, black_pixel_threshold=10
):
    """
    skip_rate is not provided via variables because this function does not
    know it is drawing object or background or an entire image
    """
    print("Skip Rate: ", skip_rate)
    img_thresh_copy = variables.img_thresh.copy()
    if object_mask is not None:
        object_mask_black_ind = np.where(object_mask == 0)
        object_ind = np.where(object_mask == 255)
        img_thresh_copy[object_mask_black_ind] = 255

    selected_ind = 0
    n_cuts_vertical = int(math.ceil(variables.resize_ht / variables.split_len))
    n_cuts_horizontal = int(math.ceil(variables.resize_wd / variables.split_len))

    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))
    print(f"Grid shape: {grid_of_cuts.shape}")

    cut_having_black = (grid_of_cuts < black_pixel_threshold) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T

    # counter for video frames, distinct from draw_cycle_counter for hand poses
    frame_write_counter = 0 
    
    while len(cut_black_indices) > 1:
        selected_ind_val = cut_black_indices[selected_ind].copy()
        range_v_start = selected_ind_val[0] * variables.split_len
        range_v_end = range_v_start + variables.split_len
        range_h_start = selected_ind_val[1] * variables.split_len
        range_h_end = range_h_start + variables.split_len

        temp_drawing_patch = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        # Ensure temp_drawing_patch is BGR for assignment
        if temp_drawing_patch.ndim == 2: # Grayscale
             temp_drawing_patch_bgr = cv2.cvtColor(temp_drawing_patch, cv2.COLOR_GRAY2BGR)
        else: # Already BGR
             temp_drawing_patch_bgr = temp_drawing_patch


        variables.drawn_frame[range_v_start:range_v_end, range_h_start:range_h_end] = temp_drawing_patch_bgr


        drawn_frame_with_hand = variables.drawn_frame.copy() # Start with current state

        if variables.hands_data: # Only draw hand if poses are loaded
            current_hand_data = variables.hands_data[variables.draw_cycle_counter % len(variables.hands_data)]
            
            # Calculate hand coordinates (center of the current drawing patch)
            hand_coord_x = range_h_start + (variables.split_len // 2) - (current_hand_data["width"] // 2)
            hand_coord_y = range_v_start + (variables.split_len // 2) - (current_hand_data["height"] // 2)
            
            drawn_frame_with_hand = draw_hand_on_img(
                drawn_frame_with_hand, # Pass the copy
                current_hand_data,
                hand_coord_x,
                hand_coord_y,
                variables.resize_ht, # Pass canvas dimensions
                variables.resize_wd
            )
            variables.draw_cycle_counter += 1 # Increment to cycle through hand poses

        cut_black_indices[selected_ind] = cut_black_indices[-1] # Efficient removal
        cut_black_indices = cut_black_indices[:-1]

        del selected_ind

        # select the next new index
        euc_arr = euc_dist(cut_black_indices, selected_ind_val)
        selected_ind = np.argmin(euc_arr)

        counter += 1
        if counter % skip_rate == 0:
            variables.video_object.write(drawn_frame_with_hand)

        if counter % 40 == 0:
            print("len of black indices: ", len(cut_black_indices))

    if object_mask is not None:
        variables.drawn_frame[:, :, :][object_ind] = variables.img[object_ind]
    else:
        variables.drawn_frame[:, :, :] = variables.img


def draw_whiteboard_animations(
    img_path, mask_path, hand_path, hand_mask_path, save_video_path, variables
):
    if mask_path is not None:
        object_mask_exists = True
    else:
        object_mask_exists = False

    # reading the image and converting it to grayscale,
    # computing clahe and later therholding
    variables = preprocess_image(img_path=img_path, variables=variables)

    # reading hand image and preprocess
    variables = preprocess_hand_image(
        hand_path=hand_path, hand_mask_path=hand_mask_path, variables=variables
    )

    # calculate how much time it takes to make video for 1 image
    start_time = time.time()

    # defining the video object
    variables.video_object = cv2.VideoWriter(
        save_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        variables.frame_rate,
        (variables.resize_wd, variables.resize_wd),
    )

    # creating an emtpy frame and select 0th index as the starting point to draw
    variables.drawn_frame = np.zeros(variables.img.shape, np.uint8) + np.array(
        [255, 255, 255], np.uint8
    )

    if object_mask_exists:

        # reading the object masks
        with open(mask_path) as file:
            object_masks = json.load(file)

        background_mask = (
            np.zeros((variables.resize_ht, variables.resize_wd), dtype=np.uint8) + 255
        )

        for object in object_masks["shapes"]:
            # Create an empty mask array
            object_mask = np.zeros((variables.img_ht, variables.img_wd), dtype=np.uint8)

            # Get the object points as a list of tuples
            object_points = np.array(object["points"], dtype=np.int32)
            object_points = np.expand_dims(object_points, axis=0)

            # Fill the polygon with white color (255) on the mask array using cv2
            cv2.fillPoly(object_mask, object_points, 255)

            # resizing the object_mask
            object_mask = cv2.resize(
                object_mask, (variables.resize_wd, variables.resize_ht)
            )

            # get the object and its background indices
            object_ind = np.where(object_mask == 255)

            # remove the object from backgrond mask
            background_mask[object_ind] = 0

            # create animation for the selected object
            draw_masked_object(
                variables=variables,
                object_mask=object_mask,
                skip_rate=variables.object_skip_rate,
            )

        # now draw the last remaing background part
        """
        # update the split len for background part by which the 
        # area covered in one loop iteration will be much larger
        """
        # Optional:
        print("Drawing the blakground region..")
        variables.split_len = 20
        draw_masked_object(
            variables=variables,
            object_mask=background_mask,
            skip_rate=variables.bg_object_skip_rate,
        )
    else:
        variables.split_len = 15
        variables.object_skip_rate = 8
        # draw the entire image without any mask
        draw_masked_object(
            variables=variables,
            skip_rate=variables.object_skip_rate,
        )

    # Ending the video with original original image
    for i in range(variables.frame_rate * variables.end_gray_img_duration_in_sec):
        variables.video_object.write(variables.img)

    # Calculating the total execution time
    end_time = time.time()
    print("total time: ", end_time - start_time)

    # closing the video object
    variables.video_object.release()


class AllVariables:
    def __init__(
        self,
        frame_rate=None,
        resize_wd=None,
        resize_ht=None,
        split_len=None,
        object_skip_rate=None,
        bg_object_skip_rate=None,
        end_gray_img_duration_in_sec=None,
    ):
        self.frame_rate = frame_rate
        self.resize_wd = resize_wd
        self.resize_ht = resize_ht
        self.split_len = split_len
        self.object_skip_rate = object_skip_rate
        self.bg_object_skip_rate = bg_object_skip_rate
        self.end_gray_img_duration_in_sec = end_gray_img_duration_in_sec


if __name__ == "__main__":
    img_path = "./images/2.png"
    save_path = "./save_videos"
    mask_path = None 
    mask_path = "./images/2.json"  # some json path
    # if no masks are available, put mask_path = None
    hand_path = "./images/drawing-hand.png"
    hand_mask_path = "./images/hand-mask.png"

    # video save path
    current_time = str(datetime.datetime.now().date())
    img_name = img_path.rsplit("/")[-1].rsplit(".")[0]
    video_save_name = img_name + "-" + current_time + ".mp4"
    save_video_path = os.path.join(save_path, video_save_name)
    print("save_video_path: ", save_video_path)

    # constants and variables object
    variables = AllVariables(
        frame_rate=25,  # frame rate for the output video
        resize_wd=1020,  # output video width
        resize_ht=1020,  # output video height
        split_len=10,  # the image is devided into grids.
        # when split_len = 10, the image is devided as: img_ht/10, img_wd/10
        object_skip_rate=8,  # when drawing,
        # 8 pixels colored will be saved together in the video
        # increase this number to make the video runtime smaller (draws faster)
        bg_object_skip_rate=14,  # assuming background region is larger,
        # hence increasing the skip rate
        end_gray_img_duration_in_sec=3,  # the last few secs of the video
        # for every image will have the entire original image shown as is
    )

    # invoking the drawing function
    draw_whiteboard_animations(
        img_path, mask_path, hand_path, hand_mask_path, save_video_path, variables
    )