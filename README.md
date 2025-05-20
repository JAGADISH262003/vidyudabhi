# AI Whiteboard Storyteller

## Overview

AI Whiteboard Storyteller is a Python project that automatically generates engaging whiteboard animation videos from a user-provided theme. It leverages Artificial Intelligence for multiple aspects of the creation process:

*   **Story Generation:** A Large Language Model (LLM) creates a narrative, scene descriptions, and narration text based on an input theme.
*   **Image Generation:** An AI image generation model creates visuals for each scene described in the story.
*   **Animation:** The script then simulates a hand drawing these images on a whiteboard and combines them with text-to-speech narration to produce a video.

The goal is to provide a tool that can quickly transform a simple idea into a dynamic visual story.

## Features

*   **Dynamic Story & Scene Generation:** Uses a Large Language Model (Meta-Llama-3-8B-Instruct via Hugging Face API) to generate a unique story, scene descriptions, and narration scripts from a given theme.
*   **AI Image Generation:** Leverages Flux AI (via Hugging Face API) to create images for each scene, with added style cues for artistic variation.
*   **Whiteboard Animation Effect:** Simulates a hand drawing each scene's image on a virtual whiteboard.
*   **Multiple Hand Poses:** Supports cycling through different hand images and masks during animation to make the drawing effect more dynamic and less repetitive.
*   **Text-to-Speech Narration:** Generates audio narration for each part of the story using Google Text-to-Speech (gTTS).
*   **Configurable Drawing Speed:** Allows users to adjust the speed of the drawing animation for objects and backgrounds.
*   **Video Concatenation:** Combines individual scene animations and their corresponding audio narrations into a single final video file with crossfade transitions.
*   **Automatic Asset Creation:** Includes helper code to create dummy hand assets if they are missing, ensuring the script is runnable out-of-the-box for demonstration.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```
(Replace `<repository_url>` and `<repository_directory>` with actual values)

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
4.  **Configuration (Optional):**
    *   **Story Theme:** You can change the theme for story generation by modifying the `story_theme` variable within the `if __name__ == "__main__":` block at the end of `integrated_storyboard_ai.py`.
        ```python
        # Example:
        story_theme = "A brave knight on a quest to find a legendary dragon."
        ```
    *   **Drawing Speed:** Adjust the drawing speed by changing `user_object_skip_rate` and `user_bg_skip_rate` variables in the same `if __name__ == "__main__":` section. Higher values result in faster drawing.
        ```python
        # Example:
        user_object_skip_rate = 10  # Faster object drawing
        user_bg_skip_rate = 20    # Faster background drawing
        ```

## Output

The script generates several files:

*   **Individual Scene Videos:** Saved in the `./save_videos/` directory, named like `image_0_YYYYMMDD_HHMMSS.mp4`, `image_1_YYYYMMDD_HHMMSS.mp4`, etc.
*   **Audio Narrations:** Individual audio files for each narration segment are saved in the `./saved_audios/` directory (e.g., `0.mp3`, `1.mp3`).
*   **Final Concatenated Video:** The complete animation with all scenes and audio is saved as `final_output.mp4` in the `./save_videos/` directory.
*   **Story Texts:**
    *   The full list of narrations is saved to `story_narrations.txt` in `./save_videos/`.
    *   The story summary is saved to `story_summary.txt` in `./save_videos/`.

## Dependencies

The project relies on the following main Python libraries:

*   **OpenCV (cv2):** For image processing, video writing, and animation tasks.
*   **NumPy:** For numerical operations, especially with image data.
*   **gTTS (Google Text-to-Speech):** For generating audio narrations from text.
*   **MoviePy:** For concatenating video clips and handling audio.
*   **Requests:** For making HTTP requests to the Hugging Face API.
*   **Pillow (PIL):** For image manipulation, particularly for creating dummy assets if needed.

---
This README should provide a comprehensive guide for users to understand, set up, and run the AI Whiteboard Storyteller project.
