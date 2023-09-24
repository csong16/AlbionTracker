import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO

# Load the model
model = YOLO('runs/detect/train4/weights/best.pt')
window_name = "Game Window"
class_names = ["DepletedSnake", "DeadSnake", "Snake", "RoughLogs", "Player", "Poacher", 
               "Toad", "DepletedRoughStone", "SoloDungeon1", "RoughStone", "BirchLogs", 
               "DepletedChestNutLogs", "DeadGiantToad", "GiantToad", "ChestnutLogs", 
               "LankyWoodpicker", "Cotton", "Flax", "SoloDungeon2", "DepletedGiantToad", 
               "LankyScavenger", "DepletedStone", "DepletedSwampSpirit", "SwampSpirit", "SoloDungeon0"]

while True:
    try:
        # Get the window
        win = gw.getWindowsWithTitle('Albion Online Client')[0]  # replace 'Your Game Window Title' with your game window's title
        
        # Get the window's location
        x, y, width, height = win.left, win.top, win.width, win.height

        # Capture the window
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        frame = np.array(screenshot)

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run the frame through the model
        results = model.track(frame)

        # Draw bounding boxes on the frame
        for result in results:
            # Get bounding boxes
            boxes = result.boxes.xyxy.numpy()  # Convert to numpy array for use with OpenCV

            # Draw bounding boxes on the frame
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_index = int(result.boxes.cls[0])
                class_name = class_names[class_index]
                label = f"Confidence: {result.boxes.conf[0]:.2f}, Class: {class_name}"
                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow(window_name, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(e)
        break

cv2.destroyAllWindows()
