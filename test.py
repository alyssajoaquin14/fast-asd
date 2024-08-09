from concurrent.futures import ThreadPoolExecutor
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def run_yolov8(file, start_frame, end_frame, model, fps, max_num_boxes):
    cap = cv2.VideoCapture(file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break

        preds = model(frame)
        # Process preds as needed and append to results
        results.append(preds)  # Store each prediction result

    cap.release()
    return results  # Return the results list

# Function to test with ThreadPoolExecutor
def test_with_thread_pool():
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submitting the run_yolov8 task to the executor
        future = executor.submit(run_yolov8, "test_vid.mp4", 0, 100, model, 2, 30)

        # Retrieving the result of the future
        result = future.result()  # This will block until the result is ready
        print("Future result:", result)

# Run the test
test_with_thread_pool()