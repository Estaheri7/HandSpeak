import os
import sys

from collections import Counter, deque
import time

import numpy as np

import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(".."))
from models.resnet import ResNet 

class HandSpeak:
    def __init__(self, model: ResNet,  device: str, class_labels: dict[int, str], transform: transforms.Compose,
                 buffer_size=20, typing_delay=1.3):
        self.device = device
        self.model = model.to(device)
        self.class_labels = class_labels
        
        # The same transform that is used in train
        self.transform = transform

        # Settings for video capture
        self.capture = cv2.VideoCapture(0)

        self.frame_buffer = deque() # Store frames
        self.buffer_size = buffer_size
        self.typed_text = ""  # Store the final predicted word
        self.last_prediction_time = time.time()  # Timer to handle typing delay
        self.typing_delay = typing_delay # Seconds to wait before adding a new letter

        self.blank = False # A flag that is True when there is a hand gesture in ROI

        # Cursor of current letter
        self.cursor_visible = True
        self.cursor_blink_interval = 0.5  # Blink interval in seconds
        self.last_cursor_blink_time = time.time()

    def video_loop(self):
        while True:
            _, frame = self.capture.read()
            frame = cv2.flip(frame, 1)  # Flip to simulate mirror image
            most_common_prediction = self.video_capture(frame)
            display_text = self.text_handler(most_common_prediction)
            
            # Display the prediction on the frame
            cv2.putText(frame, f'Prediction: {most_common_prediction}', (self.x1, self.y2+30), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Show the typed text with the cursor
            cv2.putText(frame, f'Typed Text: {display_text}', (self.x1, self.y2 + 70),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

            # Show the frame
            cv2.imshow("Frame", frame)

            # Break loop on keypress
            if cv2.waitKey(10) & 0xFF == ord('`'):
                break
        
        self.terminate_video_loop()
        
    def terminate_video_loop(self):
        self.capture.release()
        cv2.destroyAllWindows()
    
    def video_capture(self, frame):
        # Define region of interest for hand detection
        self.x1, self.y1 = 10, 10
        self.x2, self.y2 = int(0.5 * frame.shape[1]), int(0.5 * frame.shape[1])
        cv2.rectangle(frame, (self.x1-1, self.y1-1), (self.x2+1, self.y2+1), (255, 0, 0), 1)

        # Extract the ROI
        region_of_interest = frame[self.y1:self.y2, self.x1:self.x2]

        # Same filters as data creation phase
        region_of_interest = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB)
        grayscale = cv2.cvtColor(region_of_interest, cv2.COLOR_RGB2GRAY)
        guassian_blur = cv2.GaussianBlur(grayscale, (5,5), 2)
        ada_threshold = cv2.adaptiveThreshold(guassian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
        _, processed_image = cv2.threshold(ada_threshold, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Convert image to PIL format for transformations
        processed_image = Image.fromarray(processed_image)
        return self.predict(processed_image)

    def predict(self, processed_image):
        # Apply the same transformations
        processed_tensor = self.transform(processed_image).unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            output = self.model(processed_tensor)
            _, predicted = torch.max(output, 1)
            class_index = predicted.item()
            prediction = self.class_labels[class_index]

            # Add prediction to buffer
            self.frame_buffer.append(prediction)
            # Remove the first element from queue if buffer is full
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.popleft()

            # Get the most common prediction in the buffer
            most_common_prediction = Counter(self.frame_buffer).most_common(1)[0][0]

            # Change blank to True if there is no hand gesture
            if most_common_prediction == '0':
                self.blank = True
                return most_common_prediction
            # Otherwise blank is False
            self.blank = False

            return most_common_prediction

    def text_handler(self, predicted_letter):
    # Add letter to typed_text only if enough time has passed
        current_time = time.time()
        if current_time - self.last_prediction_time > self.typing_delay:
            if predicted_letter == '0':
                pass
            elif predicted_letter == 'space':
                self.typed_text += ' '
            elif predicted_letter == 'backspace':
                self.typed_text = self.typed_text[:-1]
            else:
                self.typed_text += predicted_letter
            self.last_prediction_time = current_time

        # Handle cursor visibility toggle based on time interval
        current_time = time.time()
        if current_time - self.last_cursor_blink_time > self.cursor_blink_interval:
            self.cursor_visible = not self.cursor_visible
            self.last_cursor_blink_time = current_time

        # Display the typed text along with the blinking cursor
        display_text = self.typed_text
        if self.cursor_visible:
            display_text += "|"  # Add the cursor at the end

        return display_text
