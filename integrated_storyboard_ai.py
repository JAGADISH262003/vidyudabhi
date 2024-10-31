import os
import cv2
import numpy as np
import json
import datetime
import tempfile
import requests
import io
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

# Flux AI configuration
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": "Bearer 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"}  # Replace with your actual token

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

# Function to interact with Flux AI
def query_flux_ai(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    except Exception as e:
        print(f"Error querying Flux AI: {e}")
        return None

# Function to generate images from scenes
def generate_images_from_scenes(scenes):
    images = []
    for i, scene in enumerate(scenes):
        try:
            image_bytes = query_flux_ai({"inputs": scene})
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                images.append(np.array(image))
                print(f"Image for scene {i + 1} generated")
            else:
                print(f"Error generating image for scene {i + 1}")
        except Exception as e:
            print(f"Error processing scene {i + 1}: {e}")
    return images

# Placeholder function to generate JSON data
def generate_json_data(images):
    return [{"shapes": []} for _ in images]

# Function to process and save images, JSON data, and audio files
def process_images(images, json_data, hand_path, hand_mask_path, story, summary, variables):
    audio_paths = generate_audio_files(story)  # Generate audio files

    with tempfile.TemporaryDirectory() as temp_dir:
        video_paths = []  # To store the paths of generated video files

        # Save images and JSON files to temporary directory
        for i, (image, json_content) in enumerate(zip(images, json_data)):
            try:
                img_path = os.path.join(temp_dir, f"image_{i}.png")
                json_path = os.path.join(temp_dir, f"image_{i}.json")
                
                cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                with open(json_path, 'w') as f:
                    json.dump(json_content, f)

                # Video save path
                current_time = str(datetime.datetime.now().date())
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                video_save_name = f"{img_name}-{current_time}.mp4"
                save_video_path = os.path.join(save_video_folder, video_save_name)
                video_paths.append(save_video_path)  # Store video path for later use
                print("save_video_path: ", save_video_path)

                # Call drawing function (ensure draw_whiteboard_animations exists)
                draw_whiteboard_animations(
                    img_path, json_path, hand_path, hand_mask_path, save_video_path, variables
                )
            except Exception as e:
                print(f"Error processing image {i}: {e}")

        # Save story and summary
        story_path = os.path.join(save_video_folder, "story.txt")
        summary_path = os.path.join(save_video_folder, "summary.txt")
        with open(story_path, "w") as f:
            f.write(" ".join(story))  # Join list into string
        with open(summary_path, "w") as f:
            f.write(summary)

    return story_path, summary_path, audio_paths, video_paths

# Function to handle the full whiteboard animation generation process
def generate_whiteboard_animations(scenes, story, summary):
    hand_path = "/content/drive/MyDrive/Final_Animation/assets/drawing-hand.png"
    hand_mask_path = "/content/drive/MyDrive/Final_Animation/assets/hand-mask.png"

    # Instantiate animation variables
    variables = AllVariables(
        frame_rate=25,
        resize_wd=1020,
        resize_ht=1020,
        split_len=10,
        object_skip_rate=8,
        bg_object_skip_rate=14,
        end_gray_img_duration_in_sec=3,
    )

    # Generate images using Flux AI
    images = generate_images_from_scenes(scenes)
    
    # Generate JSON data for the images
    json_data = generate_json_data(images)

    # Process images and save the story/summary
    story_path, summary_path, audio_paths, video_paths = process_images(
        images, json_data, hand_path, hand_mask_path, story, summary, variables
    )

    # Return paths to generated files
    return video_paths, audio_paths, story_path, summary_path

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
    # Example usage
    scenes = [
        "Scene 1: A young man with shoulder-length black hair, wearing a black jacket, eating a samosa, with a cafe background.",
        "Scene 2: The same young man with shoulder-length black hair and a black jacket, now holding a cup of coffee and drinking, with the same cafe background as in scene 1.",
        "Scene 3: A different person, a woman in a red dress, enters the cafe.",
        "Scene 4: The young man with shoulder-length hair glances up and smiles at her.",
        "Scene 5: The woman in the red dress orders a coffee and sits at a nearby table, glancing over at the young man.",
        "Scene 6: The young man, now curious, watches her from the corner of his eye as she settles down with her coffee.",
        "Scene 7: The woman opens a book, and the young man notices the same book he has on his table.",
        "Scene 8: The man stands up, nervously walking over to her table, holding his book in hand.",
        "Scene 9: They exchange a few words, realizing they both are reading the same book.",
        "Scene 10: The woman laughs, offering the man a seat at her table."        # Add more scenes as needed
    ]
    
    story = [
        "A story about a man's coffee shop experience.",
        "He enjoys his samosa and coffee while watching the world go by.",
        "A woman in a red dress enters the cafe.",
        "The man glances up and smiles at her.",
        "The woman orders a coffee and sits near the young man, their eyes briefly meeting.",
        "Curiosity piqued, the man discreetly watches her as she opens her book.",
        "To his surprise, the book she’s reading is the same as his.",
        "Gathering courage, the man approaches her table, showing her his copy of the book.",
        "They realize they share an interest in the same story, sparking a conversation.",
        "The woman invites the man to sit with her, and they bond over their shared love for the book."   
     ]
    summary = "Summary of the man's visit to the cafe."

    video_paths, audio_paths, story_path, summary_path = generate_whiteboard_animations(scenes, story, summary)
    
    # Concatenate videos and audios
    final_video_path = concatenate_videos_and_audios(video_paths, audio_paths)

    print("Story path:", story_path)
    print("Summary path:", summary_path)
