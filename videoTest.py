import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from moviepy.editor import VideoFileClip, ImageSequenceClip
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np

processor = AutoImageProcessor.from_pretrained("nielsr/depth-anything-small")
model = AutoModelForDepthEstimation.from_pretrained("nielsr/depth-anything-small")

net = cv2.dnn.readNet("YOLO\\yolov3.weights", "YOLO\\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("YOLO\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Process image through YOLO
def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

# Draw bounding boxes on image
def draw_boxes(img, boxes, confidences, class_ids, indexes, classes):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (255, 0, 0)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x, y - 10), f"{label} {confidence:.2f}", fill=color, font=font)
    return img

def process_image(image):

    # # Enhance Sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image_sharp = enhancer.enhance(1.5)  # Enhance sharpness by a factor of 2.0

    # image_np = np.array(image.convert('L'))  # Convert to grayscale

    # # Apply Histogram Equalization
    # equalized = cv2.equalizeHist(image_np)

    # # Convert equalized image back to PIL image
    # return_pil = Image.fromarray(equalized)

    return image_sharp

def depthDetection(img):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

    h, w = img.size[::-1]

    # interpolation back from predictions to earlier size
    depth = torch.nn.functional.interpolate(predicted_depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]

    return (Image.fromarray(colored_depth), depth)

def objectDetection(img):

    # Detect objects
    boxes, confidences, class_ids, indexes = detect_objects(img, net, output_layers)

    depth_detection = depthDetection(Image.fromarray(img))

    # Draw bounding boxes
    img_with_boxes = draw_boxes(depth_detection[0], boxes, confidences, class_ids, indexes, classes)

    # Save and show result
    return (np.array(img_with_boxes), [boxes, confidences, [classes[class_id] for class_id in class_ids], indexes])

def record_video(output_path, duration=10, fps=20.0, frame_size=(640, 480)):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    start_time = cv2.getTickCount()
    end_time = start_time + duration * cv2.getTickFrequency()

    while cv2.getTickCount() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        out.write(frame)
        cv2.imshow('Recording', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_video(input_path, output_path):
    clip = VideoFileClip(input_path)
    processed_frames = []

    all_frames = [frame for frame in clip.iter_frames()]

    total_frames = len(all_frames)

    percentCompleted = 0
    percentInterval = 0.1

    print(f"Total Frames to Process: {total_frames}")

    for i, frame in enumerate(all_frames):
        processed_frame, data = objectDetection(frame)
        processed_frames.append(processed_frame)

        if (i / total_frames) > percentCompleted:
            print(f"{percentCompleted * 100.0}% completed!")
            percentCompleted += percentInterval

    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    processed_clip.write_videofile(output_path, codec='libx264')

def display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Processed Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    NUM_SECONDS = 10

    # record_video('recorded_video.avi', duration=NUM_SECONDS)
    process_video('hotelRoomDemo.mp4', 'processed_video.mp4')
    display_video('processed_video.mp4')
