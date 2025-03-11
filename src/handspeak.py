import os
import sys

from collections import Counter, deque
import time
import tkinter as tk

import numpy as np

import cv2
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(".."))
from models.resnet import ResNet 

class HandSpeak:
    """
    A real-time American Sign Language (ASL) recognition system using a ResNet model for gesture classification.

    Attributes:
        model (ResNet or nn.Module): The pre-trained ResNet model for hand gesture recognition.
        device (str): The device to run the model on ('cpu' or 'cuda').
        class_labels (dict[int, str]): A mapping of class indices to gesture labels.
        transform (transforms.Compose): Image transformation pipeline for preprocessing input frames.
        buffer_size (int): The maximum number of predictions stored in the buffer.
        typing_delay (float): Minimum delay before registering a new letter in the typed text.
    """
    def __init__(self, model: ResNet,  device: str, class_labels: dict[int, str], transform: transforms.Compose,
                 buffer_size=20, typing_delay=1.3):
        self.device = device
        self.model = model.to(device)
        self.class_labels = class_labels
        
        # The same transform that is used in train
        self.transform = transform

        # Settings for video capture
        self.capture = cv2.VideoCapture(0)

        # ROI to create a box for hand gestures
        self.region_of_interest = None

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

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title('American Sign Language')
        self.root.geometry("800x600")
        self.root.configure(bg='#272727')
        self.root.protocol("WM_DELETE_WINDOW", self.terminate_video_loop)
        self.root.wm_attributes("-topmost", True)
        # Remove the default title bar
        self.root.overrideredirect(True)

        # Custom title bar
        self.title_bar = tk.Frame(self.root, bg='#3ba47b', relief='solid', bd=2)
        self.title_bar.pack(fill="x", side="top")

        # Exit button
        self.exit_button = tk.Button(self.title_bar, text="Exit", command=self.close_window, bg='#3ed8c3', fg='white', bd=0)
        self.exit_button.pack(side="right")

        # A label for Region of interest (which contains hand gestures)
        self.video_label = tk.Label(self.root, 
                                    bg='black', 
                                    bd=2,
                                    relief='solid',
                                    highlightthickness=3,
                                    highlightbackground="#3ba47b",
                                    highlightcolor="#3ba47b")
        self.video_label.place(x=10, y=35)

        # Label to display the most common prediction
        self.prediction_label = tk.Label(self.root, text="Prediction: ", bg='#272727', fg='#3ed8c3', font=('Arial', 28, 'bold'))
        self.prediction_label.place(x=370, y=200)

        # Label to display the typed text
        self.typed_text_label = tk.Label(self.root, text="Typed Text: ", bg='#272727', fg='#3ed8c3', font=('Arial', 24, 'bold'))
        self.typed_text_label.place(x=10, y=400)
    
    def run(self):
        """Starts the Tkinter main loop and begins video processing."""
        self.video_loop()
        self.root.mainloop()

    def close_window(self):
        """Closes the Tkinter window and stops the application."""
        self.root.quit()

    def video_loop(self):
        """Captures video frames continuously, processes them, and updates the UI."""
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip to simulate mirror image
            most_common_prediction = self.video_capture(frame)
            display_text = self.text_handler(most_common_prediction)
            self.tkinter_handler(most_common_prediction, display_text)

        # Call this function again after 10 milliseconds
        self.root.after(10, self.video_loop)
        
    def terminate_video_loop(self):
        """Releases the video capture and destroys the Tkinter window."""
        self.capture.release()
        self.root.destroy()
    
    def video_capture(self, frame):
        """Processes a video frame to extract the hand gesture region and apply transformations.
        
        Args:
            frame (np.ndarray): The captured video frame.
        
        Returns:
            str: The most common predicted gesture.
        """
        # Define region of interest for hand detection
        self.x1, self.y1 = 10, 10
        self.x2, self.y2 = int(0.5 * frame.shape[1]), int(0.5 * frame.shape[1])
        cv2.rectangle(frame, (self.x1-1, self.y1-1), (self.x2+1, self.y2+1), (255, 0, 0), 1)

        # Extract the ROI
        self.region_of_interest = frame[self.y1:self.y2, self.x1:self.x2]

        # Same filters as data creation phase
        self.region_of_interest = cv2.cvtColor(self.region_of_interest, cv2.COLOR_BGR2RGB)
        grayscale = cv2.cvtColor(self.region_of_interest, cv2.COLOR_RGB2GRAY)
        guassian_blur = cv2.GaussianBlur(grayscale, (5,5), 2)
        ada_threshold = cv2.adaptiveThreshold(guassian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
        _, processed_image = cv2.threshold(ada_threshold, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Convert image to PIL format for transformations
        processed_image = Image.fromarray(processed_image)
        return self.predict(processed_image)

    def predict(self, processed_image):
        """Runs the model prediction on the processed image.
        
        Args:
            processed_image (Image): The preprocessed hand gesture image.
        
        Returns:
            str: The most common predicted gesture from the buffer.
        """
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
        """Handles text input based on predictions and manages cursor blinking.
        
        Args:
            predicted_letter (str): The recognized gesture letter.
        
        Returns:
            str: The updated typed text with cursor display.
        """
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

    def tkinter_handler(self, predict, text):
        """Updates the Tkinter UI elements with the latest prediction and typed text.
        
        Args:
            predict (str): The most common predicted gesture.
            text (str): The currently typed text.
        """
        # preparing region of interest for tkinter
        img = Image.fromarray(self.region_of_interest)
        imgtk = ImageTk.PhotoImage(image=img)

        # put image (ROI video capture) on the root window
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # put texts on the root window
        self.prediction_label.config(text=f"Prediction: {predict}")
        self.typed_text_label.config(text=f"Typed Text: {text}")

    