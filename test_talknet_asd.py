from talknet.main import TalkNetASD

def main():
    # Initialize the TalkNetASD instance
    talknet_instance = TalkNetASD()
    talknet_instance.__setup__()

    # Define your test video file path
    test_video = "test_vid.mp4"  # Replace with the actual path to your test video

    # Set the parameters for the test
    start_time = 0
    end_time = -1  # Process until the end of the video
    return_visualization = False
    face_boxes = ""  # Provide face boxes string if available, else leave empty
    in_memory_threshold = 3000  # Adjust based on your memory constraints

    # Run the prediction
    try:
        result = talknet_instance.__predict__(
            video=test_video,
            start_time=start_time,
            end_time=end_time,
            return_visualization=return_visualization,
            face_boxes=face_boxes,
            in_memory_threshold=in_memory_threshold
        )

        # Print or save the results
        print("Prediction Result:")
        if return_visualization:
            print(f"Visualization saved at: {result}")
        else:
            print(list(result))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()