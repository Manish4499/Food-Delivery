import cv2
import numpy as np
from flask import Flask, request, render_template, Response
import os

app = Flask(__name__)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    
    # Check if slope is not zero to avoid division by zero
    if slope != 0:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    else:
        x1 = x2 = 0  # Set to 0 or some default value when slope is zero
    
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.array([0, 0], dtype=np.float64)
    right_fit_average = np.array([0, 0], dtype=np.float64)

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Change color to green (0, 255, 0)
    return line_image




def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lane_image = np.copy(frame)
        canny_image = canny(lane_image)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(lane_image, lines)
        line_image = display_lines(lane_image, averaged_lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

        ret, buffer = cv2.imencode('.jpg', combo_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_uploaded_video():
    if 'video' not in request.files:
        return "No file part"

    video_file = request.files['video']
    
    if video_file.filename == '':
        return "No selected file"

    if video_file:
        video_path = os.path.join("static/uploaded_videos", video_file.filename)
        video_file.save(video_path)
        return Response(process_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

