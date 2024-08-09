# test_yolov8.py

from yolov8_model import YOLOv8

def main():
    # Initialize the YOLOv8 class
    yolov8_instance = YOLOv8()
    
    # Call the setup method to initialize models
    yolov8_instance.__setup__()
    
    # Define a test video or image file path
    test_file = "test_vid.mp4"  # Update this with the actual path
    
    # Call the prediction method and print results
    try:
        print("Starting test prediction...")
        results = yolov8_instance.__predict__(
            file=test_file,
            confidence_threshold=0.25,
            models="yolov8l",
            start_frame=0,
            end_frame=-1,
            fps=-1,
            max_num_boxes=10
        )
        print("Prediction results:")
        print(results)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()