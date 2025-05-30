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


def preprocess_image(img_path, resize_wd, resize_ht):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    img_ht, img_wd = img.shape[0], img.shape[1]
    resized_img = cv2.resize(img, (resize_wd, resize_ht))
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # color histogram equilization (CLAHE) - cl1 is not used further, consistent with original
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cl1 = clahe.apply(img_gray)

    # gaussian adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    return {
        "original_height": img_ht,
        "original_width": img_wd,
        "processed_image_gray": img_gray,
        "image_thresh": img_thresh,
        "processed_image_color": resized_img
    }


def preprocess_hand_image(hand_path, hand_mask_path):
    hand = cv2.imread(hand_path)
    if hand is None:
        raise FileNotFoundError(f"Hand image not found at path: {hand_path}")
    hand_mask_original = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
    if hand_mask_original is None:
        raise FileNotFoundError(f"Hand mask image not found at path: {hand_mask_path}")

    top_left, bottom_right = get_extreme_coordinates(hand_mask_original)
    
    hand_cropped = hand[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    hand_mask_cropped = hand_mask_original[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    
    hand_mask_inv_cropped = 255 - hand_mask_cropped

    hand_bg_black = hand_cropped.copy()
    hand_bg_black[hand_mask_cropped == 0] = [0, 0, 0]

    hand_mask_scaled = hand_mask_cropped.astype(np.float32) / 255.0
    hand_mask_inv_scaled = hand_mask_inv_cropped.astype(np.float32) / 255.0
    
    hand_ht, hand_wd = hand_bg_black.shape[:2]

    return {
        "hand_height": hand_ht,
        "hand_width": hand_wd,
        "hand_image": hand_bg_black,
        "hand_mask_scaled": hand_mask_scaled,
        "hand_mask_inverse_scaled": hand_mask_inv_scaled
    }


def get_extreme_coordinates(mask):
    indices = np.where(mask == 255)
    # Extract the x and y coordinates of the pixels.
    x = indices[1]
    y = indices[0]

    # Find the minimum and maximum x and y coordinates.
    topleft = (np.min(x), np.min(y))
    bottomright = (np.max(x), np.max(y))

    return topleft, bottomright


def draw_hand_on_img(
    drawing,
    hand,
    drawing_coord_x,
    drawing_coord_y,
    hand_mask_inv,
    hand_ht,
    hand_wd,
    canvas_ht,  # Renamed from img_ht
    canvas_wd,  # Renamed from img_wd
):
    # Calculate the height and width of the hand that can be drawn on the canvas
    # without going out of bounds.
    remaining_ht = canvas_ht - drawing_coord_y
    remaining_wd = canvas_wd - drawing_coord_x
    
    # Determine the actual height of the hand image to crop and use.
    # If the space available on the canvas is less than the hand's height,
    # crop the hand to fit. Otherwise, use the full hand height.
    if remaining_ht > hand_ht:
        crop_hand_ht = hand_ht
    else:
        crop_hand_ht = remaining_ht

    # Determine the actual width of the hand image to crop and use.
    # If the space available on the canvas is less than the hand's width,
    # crop the hand to fit. Otherwise, use the full hand width.
    if remaining_wd > hand_wd:
        crop_hand_wd = hand_wd
    else:
        crop_hand_wd = remaining_wd

    # Crop the hand image and its inverse mask to the calculated dimensions.
    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]

    # Define the Region of Interest (ROI) on the drawing canvas where the hand will be placed.
    roi = drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ]

    # Clear the area for the hand by multiplying the ROI with the inverse hand mask.
    # This makes the pixels where the hand will be drawn black (or transparent if alpha).
    # Explicitly handle data types for each channel to prevent overflow/underflow issues.
    for i in range(3):  # Iterate over B, G, R channels
        roi_channel = roi[:, :, i].astype(np.float32)
        # The inverse mask (0 where hand is, 1 elsewhere) "erases" the part of the drawing where the hand will appear.
        masked_channel = (roi_channel * hand_mask_inv_cropped).astype(np.uint8)
        roi[:, :, i] = masked_channel
    
    # Add the cropped hand image to the ROI.
    # Since the area for the hand was cleared (made black), adding the hand image
    # places it correctly. NumPy's uint8 addition performs saturation, so values
    # will be clipped to 255 if they exceed it.
    drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ] = roi + hand_cropped # roi is already a view into drawing, direct addition is fine.
    
    return drawing


def _initialize_drawing_grid(
    base_image_thresh, object_mask, split_len, resize_ht, resize_wd, black_pixel_threshold
):
    img_thresh_copy = base_image_thresh.copy()
    object_ind = None

    if object_mask is not None:
        # Store indices where mask is 0 (background to be whitened in thresh image)
        object_mask_black_ind = np.where(object_mask == 0)
        # Store indices where mask is 255 (actual object)
        object_ind = np.where(object_mask == 255)
        img_thresh_copy[object_mask_black_ind] = 255  # Make area outside object white in threshold image

    n_cuts_vertical = int(math.ceil(resize_ht / split_len))
    n_cuts_horizontal = int(math.ceil(resize_wd / split_len))

    # Cut the image into grids
    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))
    
    # Find grids where there is at least one black pixel (these grids will be drawn)
    cut_having_black = (grid_of_cuts < black_pixel_threshold) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    # Get coordinates of these black grids
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T
    
    return img_thresh_copy, grid_of_cuts, list(cut_black_indices), object_ind


def _draw_cell_and_render_hand(
    drawn_frame, current_grid_data, cell_v_start, cell_h_start, split_len,
    hand_image, hand_mask_inverse, hand_height, hand_width, canvas_height, canvas_width
):
    range_v_end = cell_v_start + split_len
    range_h_end = cell_h_start + split_len

    # Create a temporary 3-channel image for the current grid cell
    temp_drawing = np.zeros((split_len, split_len, 3), dtype=np.uint8)
    # Assign the thresholded grid data to all channels (making it grayscale)
    temp_drawing[:, :, 0] = current_grid_data
    temp_drawing[:, :, 1] = current_grid_data
    temp_drawing[:, :, 2] = current_grid_data

    # Place this grayscale grid cell onto the main drawing frame
    drawn_frame[cell_v_start:range_v_end, cell_h_start:range_h_end] = temp_drawing

    # Calculate center coordinates for placing the hand image over this cell
    hand_coord_x = cell_h_start + int(split_len / 2)
    hand_coord_y = cell_v_start + int(split_len / 2)
    
    # Overlay the hand image
    # Pass copies if draw_hand_on_img could modify them; assuming hand_image and mask are read-only inputs to it.
    # drawn_frame.copy() is important as draw_hand_on_img modifies the frame it's given.
    drawn_frame_with_hand = draw_hand_on_img(
        drawn_frame.copy(), 
        hand_image, 
        hand_coord_x,
        hand_coord_y,
        hand_mask_inverse, 
        hand_height,
        hand_width,
        canvas_height,
        canvas_width,
    )
    return drawn_frame_with_hand


def _select_next_grid_index(
    cut_black_indices_list,  # This is a list of [row, col] arrays
    current_selected_list_idx, # Index in the list
    current_selected_coords_val # The [row, col] value at that index
):
    # Efficiently remove the processed element: swap with the last, then pop.
    cut_black_indices_list[current_selected_list_idx] = cut_black_indices_list[-1]
    cut_black_indices_list.pop()

    if not cut_black_indices_list: # If list is empty
        return None

    # Find the next closest grid cell to the one just processed
    euc_arr = euc_dist(np.array(cut_black_indices_list), current_selected_coords_val)
    next_selected_idx_in_list = np.argmin(euc_arr)
    return next_selected_idx_in_list


def _finalize_drawn_frame(drawn_frame, original_color_image, object_mask, object_ind):
    if object_mask is not None and object_ind is not None:
        # If an object mask was provided, fill in the object area with original image colors.
        # object_ind should be a tuple of arrays (e.g., from np.where) suitable for advanced indexing.
        drawn_frame[object_ind] = original_color_image[object_ind]
    else:
        # If no mask, the entire frame becomes the original image.
        drawn_frame[:, :, :] = original_color_image[:, :, :]
    # drawn_frame is modified in place.


def draw_masked_object(
    variables, object_mask=None, skip_rate=5, black_pixel_threshold=10
):
    """
    Orchestrates the drawing of masked objects by breaking down the process
    into initialization, iterative cell drawing, and finalization.
    skip_rate is not provided via variables because this function does not
    know it is drawing object or background or an entire image.
    """
    print("Skip Rate: ", skip_rate)

    # Phase 1: Initialize grid, identify cells to draw, and get object indices if mask is present
    # Note: object_mask here is the mask for the specific object being drawn,
    # not the overall background_mask which is handled by recursive calls to draw_masked_object.
    _img_thresh_copy, grid_of_cuts, cut_black_indices, object_ind = _initialize_drawing_grid(
        variables.img_thresh, # The full thresholded image of the scene
        object_mask,          # Mask for the current object (can be None)
        variables.split_len,
        variables.resize_ht,
        variables.resize_wd,
        black_pixel_threshold,
    )
    print(grid_of_cuts.shape) # Original print statement

    # If there are no "black" cells to draw for this object/image, finalize and return.
    if not cut_black_indices:
        _finalize_drawn_frame(variables.drawn_frame, variables.img, object_mask, object_ind)
        # It's important to write this frame to video if it's the only thing happening.
        # However, the main loop for writing frames is below. If this is an object,
        # it will be part of a larger scene. If it's the whole image, the
        # main draw_whiteboard_animations will handle final frame writes.
        # For now, let's assume this function's responsibility is just to update variables.drawn_frame.
        return

    # Phase 2: Iteratively draw cells
    selected_list_idx = 0  # Start with the first grid cell in the list
    counter = 0
    
    # The original loop condition was `len(cut_black_indices) > 1`.
    # This new loop continues as long as `selected_list_idx` is valid.
    # `_select_next_grid_index` will return None when no more cells are left.
    while selected_list_idx is not None:
        # Get the coordinates (e.g., [row, col] in the grid matrix) of the current cell
        current_cell_coords = cut_black_indices[selected_list_idx]

        # Calculate top-left screen coordinates for this cell
        range_v_start = current_cell_coords[0] * variables.split_len
        range_h_start = current_cell_coords[1] * variables.split_len
        
        # Get the actual image data for the current cell from the grid
        current_grid_cell_data = grid_of_cuts[current_cell_coords[0]][current_cell_coords[1]]

        # Draw the cell and the hand
        # variables.drawn_frame is modified by _draw_cell_and_render_hand
        drawn_frame_with_hand = _draw_cell_and_render_hand(
            variables.drawn_frame, 
            current_grid_cell_data,
            range_v_start,
            range_h_start,
            variables.split_len,
            variables.hand,
            variables.hand_mask_inv,
            variables.hand_ht,
            variables.hand_wd,
            variables.resize_ht, # canvas_height
            variables.resize_wd, # canvas_width
        )

        counter += 1
        if counter % skip_rate == 0:
            variables.video_object.write(drawn_frame_with_hand)

        if counter % 40 == 0:
            print("len of black indices: ", len(cut_black_indices)) # Shows remaining cells
        
        # Select the next cell to draw
        # Pass a copy of current_cell_coords as it might be modified if it's a view,
        # and cut_black_indices list is modified by the helper.
        selected_list_idx = _select_next_grid_index(
            cut_black_indices, selected_list_idx, np.array(current_cell_coords)
        )
        # Loop continues if _select_next_grid_index returns a valid index, else terminates.

    # Phase 3: Finalize the drawing (e.g., fill object with original colors)
    # This uses the object_mask and object_ind obtained from _initialize_drawing_grid.
    _finalize_drawn_frame(variables.drawn_frame, variables.img, object_mask, object_ind)
    # The very last state of variables.drawn_frame (after object color fill) 
    # might need to be written to video. This is typically handled by the
    # end_gray_img_duration_in_sec loop in the main draw_whiteboard_animations function.


def draw_whiteboard_animations(img_path, mask_path, save_video_path, variables):
    if mask_path is not None:
        object_mask_exists = True
    else:
        object_mask_exists = False

    # reading the image and converting it to grayscale,
    # computing clahe and later therholding
    image_data = preprocess_image(img_path=img_path,
                                    resize_wd=variables.resize_wd,
                                    resize_ht=variables.resize_ht)
    variables.img_ht = image_data["original_height"]
    variables.img_wd = image_data["original_width"]
    variables.img_gray = image_data["processed_image_gray"]
    variables.img_thresh = image_data["image_thresh"]
    variables.img = image_data["processed_image_color"]

    # reading hand image and preprocess
    hand_data = preprocess_hand_image(hand_path=variables.hand_path,
                                      hand_mask_path=variables.hand_mask_path)
    variables.hand_ht = hand_data["hand_height"]
    variables.hand_wd = hand_data["hand_width"]
    variables.hand = hand_data["hand_image"]
    variables.hand_mask = hand_data["hand_mask_scaled"] 
    variables.hand_mask_inv = hand_data["hand_mask_inverse_scaled"]

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
    if variables.video_object and variables.video_object.isOpened():
        variables.video_object.release()


# Helper function to set up the video writer and initial drawing frame
def _setup_video_output(save_video_path, frame_rate, resize_wd, resize_ht, initial_frame_shape_tuple, initial_bg_color_tuple=(255, 255, 255)):
    video_object = cv2.VideoWriter(
        save_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (resize_wd, resize_ht),  # Corrected dimensions
    )
    drawn_frame = np.zeros(initial_frame_shape_tuple, np.uint8) + np.array(initial_bg_color_tuple, np.uint8)
    return video_object, drawn_frame

# Helper function to load object masks from a JSON file
def _load_object_masks(mask_path):
    try:
        with open(mask_path) as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from mask file {mask_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading mask file {mask_path}: {e}")
        return None

# Helper function to process a single object from JSON mask data, draw it, and update the background mask
def _draw_single_object_and_update_background(variables, object_json_item, background_mask_accumulator, original_img_ht, original_img_wd):
    # Create an empty mask array with original image dimensions
    object_mask_single_original_size = np.zeros((original_img_ht, original_img_wd), dtype=np.uint8)
    
    # Get the object points as a list of tuples and ensure it's in the correct format for fillPoly
    object_points = np.array(object_json_item["points"], dtype=np.int32)
    if object_points.ndim == 2: # If it's a list of points, ensure it's a list of polygons for fillPoly
        object_points = [object_points]

    # Fill the polygon with white color (255) on the original size mask
    cv2.fillPoly(object_mask_single_original_size, object_points, 255)

    # Resize the object mask to the target drawing dimensions
    resized_object_mask_single = cv2.resize(
        object_mask_single_original_size, (variables.resize_wd, variables.resize_ht)
    )

    # Update the background mask accumulator: where the current object is, set background to 0 (black)
    object_indices_in_resized_mask = np.where(resized_object_mask_single == 255)
    background_mask_accumulator[object_indices_in_resized_mask] = 0

    # Call draw_masked_object for this single object
    # variables.drawn_frame will be updated by draw_masked_object
    draw_masked_object(
        variables, 
        object_mask=resized_object_mask_single, 
        skip_rate=variables.object_skip_rate
    )

# Helper function to orchestrate drawing when masks are provided
def _draw_scene_with_masks(variables, mask_path, original_img_ht, original_img_wd):
    object_masks_json = _load_object_masks(mask_path)
    if not object_masks_json or "shapes" not in object_masks_json:
        print("Mask file loaded incorrectly or has no shapes. Drawing full image.")
        # Fallback to drawing without masks if mask loading fails or is invalid
        _draw_scene_without_masks(variables)
        return

    # Initialize a background mask for the entire scene (all white initially)
    # This mask will accumulate black areas where objects are drawn.
    background_mask_for_drawing = np.zeros((variables.resize_ht, variables.resize_wd), dtype=np.uint8) + 255

    for object_item in object_masks_json["shapes"]:
        _draw_single_object_and_update_background(
            variables, 
            object_item, 
            background_mask_for_drawing, 
            original_img_ht, 
            original_img_wd
        )
    
    # After all objects are drawn, draw the remaining background
    print("Drawing the background region..")
    original_split_len = variables.split_len # Store and restore split_len if modified for background
    variables.split_len = 20 # Example: use a larger split_len for background
    draw_masked_object(
        variables, 
        object_mask=background_mask_for_drawing, 
        skip_rate=variables.bg_object_skip_rate
    )
    variables.split_len = original_split_len # Restore original split_len

# Helper function to orchestrate drawing when no masks are provided
def _draw_scene_without_masks(variables):
    print("Drawing scene without masks.")
    original_split_len = variables.split_len
    original_object_skip_rate = variables.object_skip_rate
    
    variables.split_len = 15  # Default from original code
    variables.object_skip_rate = 8 # Default from original code
    
    draw_masked_object(
        variables, 
        object_mask=None, 
        skip_rate=variables.object_skip_rate
    )
    
    variables.split_len = original_split_len # Restore
    variables.object_skip_rate = original_object_skip_rate # Restore

# Helper function to write final pause frames to the video
def _write_final_pause_frames(video_object, image_to_write, frame_rate, duration_in_seconds):
    if not video_object or not video_object.isOpened():
        print("Video writer not available for writing final frames.")
        return
    num_frames = int(frame_rate * duration_in_seconds)
    for _ in range(num_frames):
        video_object.write(image_to_write)


# Main orchestrator function
def draw_whiteboard_animations(img_path, mask_path, save_video_path, variables):
    # 1. Preprocessing
    image_data = preprocess_image(
        img_path=img_path, 
        resize_wd=variables.resize_wd, 
        resize_ht=variables.resize_ht
    )
    variables.img_ht = image_data["original_height"]    # Original image height
    variables.img_wd = image_data["original_width"]     # Original image width
    variables.img_gray = image_data["processed_image_gray"]
    variables.img_thresh = image_data["image_thresh"]
    variables.img = image_data["processed_image_color"] # Resized color image

    hand_data = preprocess_hand_image(
        hand_path=variables.hand_path, 
        hand_mask_path=variables.hand_mask_path
    )
    variables.hand_ht = hand_data["hand_height"]
    variables.hand_wd = hand_data["hand_width"]
    variables.hand = hand_data["hand_image"]
    variables.hand_mask = hand_data["hand_mask_scaled"] 
    variables.hand_mask_inv = hand_data["hand_mask_inverse_scaled"]

    start_time = time.time()
    
    # Initialize video_object to None for finally block
    variables.video_object = None 

    try:
        # 2. Setup Video Writer and initial drawn_frame
        variables.video_object, variables.drawn_frame = _setup_video_output(
            save_video_path,
            variables.frame_rate,
            variables.resize_wd,
            variables.resize_ht,
            variables.img.shape, # Shape of the resized color image
            (255, 255, 255)     # White background
        )

        # 3. Main drawing logic
        if mask_path is not None:
            _draw_scene_with_masks(variables, mask_path, variables.img_ht, variables.img_wd)
        else:
            _draw_scene_without_masks(variables)

        # 4. Write end frames (pause with original color image)
        _write_final_pause_frames(
            variables.video_object,
            variables.img, # The final color image
            variables.frame_rate,
            variables.end_gray_img_duration_in_sec
        )

    except Exception as e:
        print(f"An error occurred during whiteboard animation generation: {e}")
        # Optionally re-raise the exception if you want it to propagate
        # raise
    finally:
        # 5. Release video object
        if variables.video_object and variables.video_object.isOpened():
            variables.video_object.release()
            print("Video object released.")

    end_time = time.time()
    print("total time: ", end_time - start_time)


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
        hand_path: str = None,
        hand_mask_path: str = None,
    ):
        self.frame_rate = frame_rate
        self.resize_wd = resize_wd
        self.resize_ht = resize_ht
        self.split_len = split_len
        self.object_skip_rate = object_skip_rate
        self.bg_object_skip_rate = bg_object_skip_rate
        self.end_gray_img_duration_in_sec = end_gray_img_duration_in_sec
        self.hand_path = hand_path
        self.hand_mask_path = hand_mask_path
        # Add video_object and drawn_frame for cleaner state management within variables
        self.video_object = None
        self.drawn_frame = None


if __name__ == "__main__":
    img_path = "./images/2.png"
    save_path = "./save_videos"
    mask_path = None
    mask_path = "./images/2.json"  # some json path
    # if no masks are available, put mask_path = None

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
        hand_path="assets/drawing-hand.png",
        hand_mask_path="assets/hand-mask.png",
    )

    # invoking the drawing function
    draw_whiteboard_animations(img_path, mask_path, save_video_path, variables)