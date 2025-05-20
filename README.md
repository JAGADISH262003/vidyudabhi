# AI Educational Whiteboard Storyteller

## Overview

AI Educational Whiteboard Storyteller is a Python project that automatically generates engaging whiteboard animation videos tailored for educational content. By leveraging Artificial Intelligence, it transforms structured learning inputs into dynamic visual stories, aiming to foster creativity and provide new perspectives for students.

The process involves:

*   **Educational Script Generation:** A Large Language Model (LLM) crafts a narrative, scene descriptions, and narration text based on user-defined educational parameters such as topic, learning objectives, and target grade level.
*   **AI Image Generation:** An AI image generation model creates visuals specifically designed for clarity and educational impact for each scene.
*   **Whiteboard Animation:** The script simulates a hand drawing these images on a whiteboard and combines them with text-to-speech narration to produce a complete educational video.

The goal is to provide educators and content creators with a tool that can quickly convert educational concepts into engaging video content.

## Features

*   **Structured Educational Script Generation:** Uses a Large Language Model (Meta-Llama-3-8B-Instruct via Hugging Face API) to generate a unique educational script (summary, scene descriptions, narrations) based on defined inputs like topic, learning objectives, target grade level, key concepts, and a creative angle.
*   **Education-Focused AI Image Generation:** Leverages Flux AI (via Hugging Face API) to create images for each scene, utilizing style cues optimized for educational clarity and visual appeal.
*   **Whiteboard Animation Effect:** Simulates a hand drawing each scene's image on a virtual whiteboard.
*   **Multiple Hand Poses:** Supports cycling through different hand images and masks during animation to make the drawing effect more dynamic.
*   **Text-to-Speech Narration:** Generates audio narration for each part of the story using Google Text-to-Speech (gTTS).
*   **Configurable Drawing Speed:** Allows users to adjust the speed of the drawing animation for objects and backgrounds.
*   **Script Output for Review:** Generates a human-readable text file (`educational_script.txt`) containing the input parameters and the full generated script (summary, scenes, and narrations) before video rendering, allowing for review.
*   **Video Concatenation:** Combines individual scene animations and their corresponding audio narrations into a single final video file with crossfade transitions.
*   **Automatic Asset Creation:** Includes helper code to create dummy hand assets if they are missing, ensuring the script is runnable out-of-the-box for demonstration.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```
(Replace `<repository_url>` and `<repository_directory>` with actual values if you are hosting this project)

### 2. Python Version
Ensure you have Python 3.8 or newer installed.

### 3. Install Dependencies
Install the necessary Python libraries using pip:
```bash
pip install opencv-python numpy gtts moviepy requests Pillow
```

### 4. API Key Configuration (Hugging Face)
This project requires a Hugging Face API token to use the LLM for story generation and the Flux AI model for image generation.

*   Obtain an API token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).
*   Set this token as an environment variable named `HF_API_TOKEN`.
    *   On Linux/macOS: `export HF_API_TOKEN='your_token_here'`
    *   On Windows (PowerShell): `$env:HF_API_TOKEN='your_token_here'`
    *   Alternatively, you can set it system-wide or within your IDE's environment configuration.

### 5. Hand Poses for Animation
The animation uses images of a hand and corresponding masks to simulate drawing.

*   Hand pose images and their masks should be placed in the `./assets/hand_poses/` directory.
*   **Naming Convention:**
    *   Hand images: `hand_pose_0.png`, `hand_pose_1.png`, `hand_pose_2.png`, etc.
    *   Corresponding masks: `hand_pose_0_mask.png`, `hand_pose_1_mask.png`, `hand_pose_2_mask.png`, etc.
    The number in the filename determines the order.
*   **Fallback:** If the `./assets/hand_poses/` directory is empty or pose files are misconfigured (e.g., a hand image is present but its mask is missing), the script will fall back to using a default single hand image (`./assets/drawing-hand.png`) and mask (`./assets/hand-mask.png`). A warning will be printed if this occurs.
*   **Dummy Asset Creation:** For initial runnability, if the default hand assets (`./assets/drawing-hand.png`, `./assets/hand-mask.png`) or the hand pose directory/files are missing, the script will attempt to create basic dummy versions by copying available assets or creating placeholder images.

## How to Run

1.  **Ensure Setup is Complete:** Verify all dependencies are installed and the `HF_API_TOKEN` is set.
2.  **Navigate to Script Directory:** Open your terminal or command prompt and navigate to the project's root directory.
3.  **Run the Script:**
    ```bash
    python integrated_storyboard_ai.py
    ```
4.  **Configuration (Crucial for Educational Content):**
    The core of the educational content generation is controlled by the `educational_parameters` dictionary within the `if __name__ == "__main__":` block at the end of `integrated_storyboard_ai.py`. Modify this dictionary to define your educational video:

    ```python
    educational_parameters = {
        # Topic: The main subject of the educational video. (string)
        "topic": "Photosynthesis",
        
        # Learning Objectives: What the viewer should understand after watching. (list of strings)
        "learning_objectives": [
            "Understand how plants make their own food using sunlight, water, and carbon dioxide.",
            "Identify the key inputs (sunlight, water, CO2) and outputs (glucose, oxygen) of photosynthesis.",
            "Recognize the role of chlorophyll in capturing light energy."
        ],
        
        # Target Grade Level: The intended audience's grade level. (string, e.g., "3rd Grade", "Middle School", "High School Biology")
        "target_grade_level": "5th Grade",
        
        # Key Concepts: Specific terms or ideas to be highlighted. (list of strings, optional, can be an empty list: [])
        "key_concepts": ["chlorophyll", "glucose (sugar)", "stomata", "carbon dioxide", "oxygen"],
        
        # Creative Angle: A specific theme or storytelling approach. (string, optional, can be an empty string: "")
        "creative_angle": "Imagine plants as tiny, super-efficient solar-powered food factories!"
    }
    ```

    *   **Drawing Speed:** Adjust the drawing speed by changing `user_object_skip_rate` and `user_bg_skip_rate` variables in the same `if __name__ == "__main__":` section. Higher values result in faster drawing.
        ```python
        user_object_skip_rate = 8   # Default: 8
        user_bg_skip_rate = 14      # Default: 14
        ```

## Output

The script generates several files, primarily in the `./save_videos/` and `./saved_audios/` directories:

*   **Educational Script (`educational_script.txt`):** Saved in `./save_videos/`. Contains the input parameters (topic, objectives, etc.) and the full LLM-generated script (summary, scene descriptions, and narrations). This allows for review before or after video generation.
*   **Individual Scene Videos:** Saved in the `./save_videos/` directory, named like `image_0_YYYYMMDD_HHMMSS.mp4`, etc.
*   **Audio Narrations:** Individual audio files for each narration segment are saved in the `./saved_audios/` directory (e.g., `0.mp3`, `1.mp3`).
*   **Final Concatenated Video:** The complete animation with all scenes and audio is saved as `final_output.mp4` in the `./save_videos/` directory.
*   **Story Texts (Legacy):**
    *   `story_narrations.txt`: The full list of narrations (also part of `educational_script.txt`).
    *   `story_summary.txt`: The story summary (also part of `educational_script.txt`).

## Dependencies

The project relies on the following main Python libraries:

*   **OpenCV (cv2):** For image processing, video writing, and animation tasks.
*   **NumPy:** For numerical operations, especially with image data.
*   **gTTS (Google Text-to-Speech):** For generating audio narrations from text.
*   **MoviePy:** For concatenating video clips and handling audio.
*   **Requests:** For making HTTP requests to the Hugging Face API.
*   **Pillow (PIL):** For image manipulation, particularly for creating dummy assets if needed.

---
This README aims to provide a comprehensive guide for users to understand, set up, and run the AI Educational Whiteboard Storyteller project.The `README.md` file has been successfully updated in the previous turn (Turn 21).

Key changes made:
1.  **Project Title:** Updated to "AI Educational Whiteboard Storyteller".
2.  **Overview:** Rewritten to emphasize educational video generation from structured learning inputs and the goal of fostering creativity for students.
3.  **Features:**
    *   "Dynamic Story & Scene Generation" was updated to "Structured Educational Script Generation," highlighting the use of inputs like topic, learning objectives, grade level, etc.
    *   A new feature, "Script Output for Review," was added, describing the `educational_script.txt` file.
    *   "AI Image Generation" was updated to "Education-Focused AI Image Generation" to reflect the style cue changes.
4.  **How to Run:**
    *   The "Configuration" subsection was significantly updated to explain the new `educational_parameters` dictionary.
    *   Each key within `educational_parameters` (`topic`, `learning_objectives`, `target_grade_level`, `key_concepts`, `creative_angle`) is now listed and described with examples and notes on optional fields.
5.  **Output:**
    *   `educational_script.txt` was added to the list of output files, with a description of its content.
    *   The existing `story_narrations.txt` and `story_summary.txt` were marked as "(Legacy)" and noted that their content is also part of `educational_script.txt`.

The Setup Instructions (API Key, Hand Poses) and Dependencies sections were already accurate and did not require changes. The language is clear and reflects the current state of the project.
