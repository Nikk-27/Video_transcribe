from flask import Flask, render_template, redirect
import gradio as gr
import whisper
import os
import subprocess
from whisper.utils import write_vtt

# Set up Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gradio')
def gradio_interface():
    return redirect("http://127.0.0.1:7860")  # Redirect to the Gradio interface

# Full path to the ffmpeg executable
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# Set the environment variable for ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# Function to transcribe and translate video using whisper
model = whisper.load_model("medium")

def check_ffmpeg():
    try:
        subprocess.run([FFMPEG_PATH, "-version"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr}")
        raise
    except FileNotFoundError:
        raise FileNotFoundError(f"ffmpeg is not found at {FFMPEG_PATH}. Please ensure it is installed and the path is correct.")

check_ffmpeg()  # Ensure ffmpeg is available

def video2mp3(video_file, output_ext="mp3"):
    print(f"video_file: {video_file}")  # Debugging statement
    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"The video file {video_file} does not exist.")
    
    filename, ext = os.path.splitext(video_file)
    try:
        subprocess.run(
            [FFMPEG_PATH, "-y", "-i", video_file, f"{filename}.{output_ext}"],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error while converting video to MP3: {e.stderr}")
        raise
    return f"{filename}.{output_ext}"

def translate(input_video):
    if input_video is None:
        raise ValueError("No input video provided.")
    
    print(f"Received input_video: {input_video}")  # Debugging statement
    
    audio_file = video2mp3(input_video)
    
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(audio_file, **translate_options)

    output_dir = os.getcwd()
    audio_path = os.path.splitext(os.path.basename(audio_file))[0]
    vtt_file = f"{audio_path}.vtt"
    print("**", vtt_file)

    # vtt_file = os.path.join(output_dir, f"{audio_path}.vtt")
    with open(vtt_file, "w") as vtt:
        write_vtt(result["segments"], file=vtt)

    subtitle = vtt_file  # Use the generated VTT file
    output_video = os.path.join(output_dir, f"{audio_path}subtitled.mp4")

    print(f"Subtitle file path: {subtitle}")  # Debugging statement
    print(f"Output video path: {output_video}")  # Debugging statement

    try:
        # Use double quotes around the subtitle path within the filter argument
        subprocess.run(
            [FFMPEG_PATH, "-i", input_video, "-vf", f"subtitles={subtitle}", output_video, '-report'],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error while adding subtitles to video: {e.stderr}")
        raise

    return output_video

title = "Add Text/Caption to your YouTube Shorts - MultiLingual"

# Create Gradio Interface
def launch_interface():
    iface = gr.Interface(
        fn=translate,
        inputs=gr.Video(label="Input Video"),
        outputs=gr.Video(label="Output Video"),
        title=title,
        live=True
    )
    iface.launch()
    
if __name__ == "__main__":
    launch_interface()
    app.run(debug=True, port=5000)
