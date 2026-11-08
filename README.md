
# Football Players Tracking with YOLOv8 and ByteTrack

## 1. Project Overview

This project develops a computer vision system for detecting
and tracking objects in football match videos.

The system uses YOLOv8 for object detection and ByteTrack
for player tracking.

Four object classes are supported:
- Player
- Goalkeeper
- Referee
- Ball

The system produces an annotated video and a CSV file
containing detection and tracking results.

## 2. Technologies

- Python
- Ultralytics YOLOv8
- ByteTrack
- OpenCV
- PyTorch
- NumPy
- Pandas

## 3. Project Structure

football-player-tracking/
- src/                 Python source code
- configs/             ByteTrack configurations
- data/                Training and evaluation datasets
- videos/              Input football videos
- outputs/
  - training/          Trained models and experiment results
  - final_demo/        Final video and tracking CSV
  - xai/               Grad-CAM visualizations
- report/
  - final_report.qmd   Quarto report source
  - final_report.pdf   Final report
  - references.bib     Academic references
- requirements.txt     Python dependencies
- README.md            Project documentation

## 4. Installation

Open PowerShell in the project directory.

Create a virtual environment if needed:

    python -m venv .venv

Activate the environment:

    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r requirements.txt

## 5. Running the Final Demo

Main script:

    src/09_final_demo.py

Example command:

    python src/09_final_demo.py --input videos/match_stable.mp4 --seconds 2

The --seconds parameter limits processing duration.
Use --seconds 0 to process the full input video.

Note: Running the script can overwrite existing output
files with the same names. Back up important results first.

## 6. Model and Tracker

The final demonstration uses the trained YOLOv8n checkpoint:

    outputs/training/yolov8n_100epochs/weights/best.pt

ByteTrack configuration:

    configs/bytetrack_v3.yaml

Separate Group Split experiments compare YOLOv8n and YOLOv8s.
Their evaluation checkpoints are not the same as the
YOLOv8n checkpoint used for the final demonstration.

## 7. Output Files

The supplied final demonstration files are:

    outputs/final_demo/match_roboflow_final.mp4
    outputs/final_demo/match_roboflow_final.csv

CSV fields include:
- frame
- timestamp
- class_id
- class_name
- track_id
- confidence
- x1, y1, x2, y2

Track IDs are assigned to players.
Other detected classes do not receive persistent Track IDs
in the final demo.

## 8. Evaluation

The project evaluates YOLOv8 detection using metrics
such as Precision, Recall, mAP@50 and mAP@50-95.

YOLOv8n and YOLOv8s are compared on a Group Split test set.

Grad-CAM visualizations are used to investigate
model behavior.

## 9. Limitations

- Small football objects can be difficult to detect.
- Player and Referee may occasionally be confused.
- Occlusion and rapid motion can affect tracking stability.
- A Track ID does not necessarily represent one unique
  real-world player across the entire video.
- Formal tracking metrics require tracking ground truth.

## 10. Report

The academic report is located at:

    report/final_report.pdf

Quarto source:

    report/final_report.qmd

To regenerate the PDF with Quarto installed:

    quarto render report/final_report.qmd --to pdf

## 11. Online Web Application

The Football AI Tracking application is deployed using
Streamlit Community Cloud.

Web App URL:
https://football-ai-tracking.streamlit.app/

GitHub Repository:
https://github.com/huynhhiep254/football-ai-tracking-webapp

### How to use the application

1. Open the Web App URL.
2. Upload an MP4 football video.
3. Select the analysis duration (2, 5, or 10 seconds).
4. Click "Run YOLOv8 + ByteTrack".
5. View the processed tracking video.
6. Download the results in MP4 and CSV formats.

The application detects Player, Goalkeeper, Referee,
and Ball. Persistent tracking IDs are assigned to
detected players.

The model may occasionally confuse players and referees,
and small football objects can be difficult to detect.
