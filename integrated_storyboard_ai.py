import os
import cv2
import numpy as np
import json
import datetime
import tempfile
import requests
import io
import random # Added for style cue selection
import glob # For finding hand pose files
import shutil # For copying files
from PIL import Image
from draw_whiteboard_animations import draw_whiteboard_animations
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

# Set save folders for audio and video
save_audio_folder = "./saved_audios"
save_video_folder = "./save_videos"
os.makedirs(save_audio_folder, exist_ok=True)
os.makedirs(save_video_folder, exist_ok=True)

# Function to generate audio for each story segment
def generate_audio_files(story):
    audio_paths = []
    for i, segment in enumerate(story):
        try:
            tts = gTTS(segment)
            audio_path = os.path.join(save_audio_folder, f'{i}.mp3')
            tts.save(audio_path)
            audio_paths.append(audio_path)
        except Exception as e:
            print(f"Error generating audio for segment {i}: {e}")
    return audio_paths

# AI Model Configurations
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # Or any other suitable model

# Style Cues for Image Generation
STYLE_CUES = [
    "cinematic lighting", "digital painting", "concept art", "cartoon style", 
    "photorealistic", "impressionistic", "watercolor style", "line art with color wash",
    "sci-fi art", "fantasy art", "steampunk style", "vintage photography", "minimalist"
]
QUALITY_ENHANCER = "detailed, high quality, sharp focus"
NEGATIVE_PROMPT = "blurry, deformed, watermark, text, low quality, artifacts, noise, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, signature, NFixer, NsfwExplicit"

# Class for holding animation variables
class AllVariables:
    def __init__(self, frame_rate, resize_wd, resize_ht, split_len, object_skip_rate, bg_object_skip_rate, end_gray_img_duration_in_sec):
        self.frame_rate = frame_rate
        self.resize_wd = resize_wd
        self.resize_ht = resize_ht
        self.split_len = split_len
        self.object_skip_rate = object_skip_rate
        self.bg_object_skip_rate = bg_object_skip_rate
        self.end_gray_img_duration_in_sec = end_gray_img_duration_in_sec

# Function to interact with Flux AI (Image Generation)
def query_flux_ai(payload):
    try:
        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face API token not found. Set the HF_API_TOKEN environment variable.")
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        response = requests.post(IMAGE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error querying Image Generation AI: {e}")
        return None
    except ValueError as e:
        print(e)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during image generation query: {e}")
        return None

# Function to query Text Generation LLM
def query_text_generation_ai(payload, model_id):
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    try:
        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face API token not found. Set the HF_API_TOKEN environment variable.")
        
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Text Generation AI ({model_id}): {e}")
        if response is not None:
            print(f"Response content: {response.text}")
        return None
    except ValueError as e:
        print(e)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during text generation query ({model_id}): {e}")
        return None

# Function to generate story from theme using LLM
def generate_story_from_theme(theme: str) -> tuple[list[str], list[str], str]:
    prompt = f"""
You are a creative storyteller. Generate a short story based on the theme: '{theme}'.
Provide the output in JSON format with the following keys:
- "summary": A brief summary of the story (50-100 words).
- "scenes": A list of 5 to 10 short scene descriptions (max 20-25 words each). These scenes should visually represent the story's progression.
- "narrations": A list of narration texts, corresponding to each scene (max 30-40 words each).

Example Theme: "A cat discovers a magical hat."
Example JSON Output:
{{
  "summary": "Whiskers, a curious cat, finds a magical hat that grants him the ability to talk to other animals. He uses his new gift to help his friends and learns the true meaning of communication.",
  "scenes": [
    "A tabby cat curiously batting at a sparkling top hat in an attic.",
    "The cat wearing the hat, looking surprised as a mouse talks to it.",
    "The cat mediating a dispute between two squirrels.",
    "The cat and a dog sharing a friendly conversation under a tree.",
    "The cat looking content, surrounded by various animal friends."
  ],
  "narrations": [
    "In a dusty attic, Whiskers stumbled upon a hat shimmering with faint magic.",
    "The moment the hat touched his head, the chattering of a tiny mouse became clear words!",
    "He soon found himself a surprising diplomat, solving quarrels in the animal kingdom.",
    "Friendships blossomed in the most unexpected places, all thanks to the chatty hat.",
    "Whiskers learned that understanding one another was the greatest magic of all."
  ]
}}
"""
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.7, "max_new_tokens": 1024, "return_full_text": False} # Added return_full_text
    }
    
    try:
        response_data = query_text_generation_ai(payload, LLM_MODEL_ID)
        if response_data and isinstance(response_data, list) and response_data[0] and "generated_text" in response_data[0]:
            # The actual JSON string is within 'generated_text'
            json_string = response_data[0]["generated_text"]
            # It seems Llama models sometimes add the prompt to the generated_text, try to remove it.
            # A simple way is to find the first '{' if the prompt is also included.
            json_start_index = json_string.find('{')
            if json_start_index != -1:
                json_string = json_string[json_start_index:]

            story_data = json.loads(json_string)
            
            summary = story_data.get("summary", "")
            scenes = story_data.get("scenes", [])
            narrations = story_data.get("narrations", [])

            if not all([summary, scenes, narrations]):
                print("Warning: LLM response missing some fields (summary, scenes, or narrations).")
                return [], [], ""
            if len(scenes) != len(narrations):
                print("Warning: Mismatch between number of scenes and narrations from LLM.")
                # Attempt to use the minimum length
                min_len = min(len(scenes), len(narrations))
                scenes = scenes[:min_len]
                narrations = narrations[:min_len]
                if not scenes: # if min_len was 0
                     return [], [], ""

            return scenes, narrations, summary
        else:
            print("Error: Unexpected response format from LLM or no generated text.")
            if response_data:
                 print(f"LLM raw response: {response_data}")
            return [], [], ""
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response from LLM: {e}")
        if response_data and isinstance(response_data, list) and response_data[0] and "generated_text" in response_data[0]:
             print(f"LLM raw text that failed parsing: {response_data[0]['generated_text']}")
        return [], [], ""
    except Exception as e:
        print(f"An unexpected error occurred in generate_story_from_theme: {e}")
        return [], [], ""

# Function to generate images from scenes
def generate_images_from_scenes(scenes: list[str]):
    images = []
    if not scenes:
        print("No scenes provided to generate images.")
        return images

    for i, original_scene_prompt in enumerate(scenes):
        try:
            # Randomly select 1 or 2 style cues
            num_styles_to_select = random.randint(1, 2)
            selected_styles = random.sample(STYLE_CUES, num_styles_to_select)
            style_string = ", ".join(selected_styles)

            # Construct the enhanced prompt
            enhanced_prompt = f"{original_scene_prompt}, {style_string}, {QUALITY_ENHANCER}"
            
            print(f"Generating image for scene {i+1} with prompt: \"{enhanced_prompt}\"")
            
            payload = {
                "inputs": enhanced_prompt,
                "negative_prompt": NEGATIVE_PROMPT 
            }
            
            image_bytes = query_flux_ai(payload)
            
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                images.append(np.array(image))
                print(f"Image for scene {i + 1} generated successfully.")
            else:
                print(f"Error generating image for scene {i + 1}: No image data returned.")
        except Exception as e:
            print(f"Error processing scene {i + 1} ('{original_scene_prompt}'): {e}")
    return images

# Placeholder function to generate JSON data (if needed for whiteboard animation)
# For now, it seems draw_whiteboard_animations might not strictly need complex shape data if we're just animating full images.
# If it does, this function might need to be more sophisticated or be removed if not used.
def generate_json_data(images):
    # Assuming basic animation of the whole image, so no specific shapes needed.
    # The draw_whiteboard_animations function might need adjustment if this is the case.
    return [{"shapes": []} for _ in images] # Returns empty shapes, assuming full image animation

# Function to process and save images, JSON data, and audio files
def process_images(images, json_data, story_narrations, story_summary, variables): # Removed hand_path, hand_mask_path
    audio_paths = generate_audio_files(story_narrations)

    hand_poses_dir = "./assets/hand_poses/"
    default_hand_image = "./assets/drawing-hand.png"
    default_hand_mask = "./assets/hand-mask.png"

    hand_pose_paths_list = sorted(glob.glob(os.path.join(hand_poses_dir, "hand_pose_*.png")))
    # Derive mask paths from image paths, assuming specific naming convention.
    hand_mask_pose_paths_list = []
    for p_path in hand_pose_paths_list:
        # Expecting mask name like hand_pose_0_mask.png from hand_pose_0.png
        mask_name = os.path.basename(p_path).replace(".png", "_mask.png")
        hand_mask_pose_paths_list.append(os.path.join(hand_poses_dir, mask_name))

    # Validate that each hand image has a corresponding mask file
    valid_hand_poses = []
    valid_hand_mask_poses = []
    for img_path, mask_path in zip(hand_pose_paths_list, hand_mask_pose_paths_list):
        if os.path.exists(mask_path):
            valid_hand_poses.append(img_path)
            valid_hand_mask_poses.append(mask_path)
        else:
            print(f"Warning: Mask file {mask_path} not found for hand image {img_path}. Skipping this pose.")
            
    hand_pose_paths_list = valid_hand_poses
    hand_mask_pose_paths_list = valid_hand_mask_poses

    if not hand_pose_paths_list:
        print(f"Warning: No hand poses found in {hand_poses_dir}. Falling back to default hand images.")
        if os.path.exists(default_hand_image) and os.path.exists(default_hand_mask):
            hand_pose_paths_list = [default_hand_image]
            hand_mask_pose_paths_list = [default_hand_mask]
        else:
            print(f"Error: Default hand images not found at {default_hand_image} or {default_hand_mask}. Hand animation will likely fail.")
            hand_pose_paths_list = [] # Ensure lists are empty if defaults also missing
            hand_mask_pose_paths_list = []
    else:
        print(f"Found {len(hand_pose_paths_list)} hand poses in {hand_poses_dir}.")


    with tempfile.TemporaryDirectory() as temp_dir:
        video_paths = []

        if not images:
            print("No images to process.")
            return "", "", [], []

        for i, (image, json_content) in enumerate(zip(images, json_data)):
            try:
                img_path = os.path.join(temp_dir, f"image_{i}.png")
                json_path = os.path.join(temp_dir, f"image_{i}.json")
                
                cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                with open(json_path, 'w') as f:
                    json.dump(json_content, f)

                current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) # More precise timestamp
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                video_save_name = f"{img_name}_{current_time}.mp4" # Use underscore for readability
                save_video_path = os.path.join(save_video_folder, video_save_name)
                video_paths.append(save_video_path)
                print(f"Video for image {i} will be saved to: {save_video_path}")

                # Call drawing function with lists of hand paths
                draw_whiteboard_animations(
                    img_path, json_path, hand_pose_paths_list, hand_mask_pose_paths_list, save_video_path, variables
                )
            except Exception as e:
                print(f"Error processing image {i} for video generation: {e}")
        
        final_story_path = ""
        final_summary_path = ""
        if story_narrations:
            final_story_path = os.path.join(save_video_folder, "story_narrations.txt")
            with open(final_story_path, "w") as f:
                f.write("\n".join(story_narrations)) # Save each narration on a new line
        
        if story_summary:
            final_summary_path = os.path.join(save_video_folder, "story_summary.txt")
            with open(final_summary_path, "w") as f:
                f.write(story_summary)

    return final_story_path, final_summary_path, audio_paths, video_paths

# Function to handle the full whiteboard animation generation process
def generate_whiteboard_animations(theme: str, object_skip_rate: int, bg_object_skip_rate: int):
    print(f"Starting whiteboard animation generation for theme: '{theme}'")
    print(f"Using Object Skip Rate: {object_skip_rate}, Background Skip Rate: {bg_object_skip_rate}")

    scenes_list, narrations_list, story_summary = generate_story_from_theme(theme)

    if not scenes_list or not narrations_list:
        print("Story generation failed or returned empty. Aborting animation process.")
        return [], [], "", ""

    print(f"Story Summary: {story_summary}")
    print(f"Generated {len(scenes_list)} scenes.")
    print(f"Generated {len(narrations_list)} narrations.")

    variables = AllVariables(
        frame_rate=25, 
        resize_wd=1020, 
        resize_ht=1020, 
        split_len=10, 
        object_skip_rate=object_skip_rate, # Use parameter value
        bg_object_skip_rate=bg_object_skip_rate, # Use parameter value
        end_gray_img_duration_in_sec=3 
    )

    images = generate_images_from_scenes(scenes_list)
    if not images:
        print("Image generation failed or returned no images. Aborting animation process.")
        return [], [], story_summary, "" # Return summary and empty paths

    json_data = generate_json_data(images) # Generate basic JSON data for each image

    # Pass narrations_list as story_narrations and story_summary to process_images
    # hand_path and hand_mask_path are no longer passed here, process_images handles discovery
    processed_story_path, processed_summary_path, audio_paths, video_paths = process_images(
        images, json_data, narrations_list, story_summary, variables
    )

    return video_paths, audio_paths, processed_story_path, processed_summary_path

# Function to concatenate videos and audios
def concatenate_videos_and_audios(video_paths, audio_paths):
    final_clips = []
    
    # Ensure that the number of video clips and audio clips match
    if len(video_paths) != len(audio_paths):
        print("Warning: Number of video clips and audio clips do not match.")
    
    for i in range(min(len(video_paths), len(audio_paths))):  # Loop over the minimum of the two lengths
        video_path = video_paths[i]
        audio_path = audio_paths[i]
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)

            # Set the audio for the video clip
            video_clip = video_clip.set_audio(audio_clip)
            final_clips.append(video_clip)
        except Exception as e:
            print(f"Error processing video/audio {video_path}: {e}")

    # Concatenate all video clips with a crossfade effect
    try:
        # Create a list to hold clips with transitions
        transition_duration = 1  # Duration for the crossfade (adjust as needed)
        
        final_video = final_clips[0]  # Start with the first clip
        for clip in final_clips[1:]:
            # Apply crossfade effect
            final_video = concatenate_videoclips([final_video, clip.crossfadein(transition_duration)], method="compose")

        # Write the final video to a file
        final_output_path = os.path.join(save_video_folder, "final_output.mp4")
        final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

        # Close all video clips
        for clip in final_clips:
            clip.close()

        print("Final video saved at:", final_output_path)
        return final_output_path
    except Exception as e:
        print(f"Error during video concatenation: {e}")
        return None

if __name__ == "__main__":
    # Define the theme for the story
    story_theme = "A curious robot discovers a hidden garden in a post-apocalyptic city"
    # story_theme = "A lost star finds its way back to its constellation with the help of a wise old owl."
    # story_theme = "A child who can talk to plants and helps a wilting forest recover."

    # User-configurable drawing speeds
    user_object_skip_rate = 8   # Default: 8. Higher values draw faster.
    user_bg_skip_rate = 14      # Default: 14. Higher values draw faster.

    # Base asset directory
    assets_dir = "./assets/"
    hand_poses_dir = os.path.join(assets_dir, "hand_poses/")
    os.makedirs(hand_poses_dir, exist_ok=True)

    # Original hand image paths (used as source for dummy poses)
    original_hand_image = os.path.join(assets_dir, "drawing-hand.png")
    original_hand_mask = os.path.join(assets_dir, "hand-mask.png")

    # Create dummy original hand images if they don't exist (for first-time run or clean env)
    if not os.path.exists(original_hand_image):
        try:
            from PIL import Image as PImage # Renamed to avoid conflict with 'Image' from global imports
            dummy_img = PImage.new('RGBA', (100, 100), (255, 0, 0, 0)) # Transparent red square
            dummy_img.save(original_hand_image)
            print(f"Created dummy original hand image at {original_hand_image}")
        except Exception as e:
            print(f"Could not create dummy original hand image: {e}")
            
    if not os.path.exists(original_hand_mask):
        try:
            from PIL import Image as PImage
            dummy_img = PImage.new('RGBA', (100, 100), (0, 0, 0, 255)) # Opaque black square
            dummy_img.save(original_hand_mask)
            print(f"Created dummy original hand mask at {original_hand_mask}")
        except Exception as e:
            print(f"Could not create dummy original hand mask: {e}")

    # Create dummy hand pose files by copying originals if they don't exist
    dummy_poses_to_create = {
        "hand_pose_0.png": original_hand_image,
        "hand_pose_0_mask.png": original_hand_mask,
        "hand_pose_1.png": original_hand_image, # Using original again for variety
        "hand_pose_1_mask.png": original_hand_mask 
    }

    for pose_file, source_file in dummy_poses_to_create.items():
        dest_path = os.path.join(hand_poses_dir, pose_file)
        if not os.path.exists(dest_path):
            if os.path.exists(source_file):
                try:
                    shutil.copy(source_file, dest_path)
                    print(f"Created dummy pose file: {dest_path} from {source_file}")
                except Exception as e:
                    print(f"Error copying {source_file} to {dest_path}: {e}")
            else:
                print(f"Warning: Source file {source_file} for dummy pose does not exist. Cannot create {dest_path}.")
    
    # Generate the whiteboard animation based on the theme
    # Hand paths are no longer passed here; process_images will discover them
    video_paths, audio_paths, story_file_path, summary_file_path = generate_whiteboard_animations(
        story_theme, user_object_skip_rate, user_bg_skip_rate
    )

    if video_paths and audio_paths:
        # Concatenate videos and audios
        final_video_path = concatenate_videos_and_audios(video_paths, audio_paths)
        if final_video_path:
            print("Final video generated successfully:", final_video_path)
        else:
            print("Failed to generate the final concatenated video.")
    else:
        print("No videos or audios were generated to concatenate.")

    if story_file_path:
        print("Story narrations saved at:", story_file_path)
    if summary_file_path:
        print("Story summary saved at:", summary_file_path)
