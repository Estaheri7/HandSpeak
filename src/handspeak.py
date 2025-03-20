import os
import sys
import json

from collections import Counter, deque
import time
import tkinter as tk

import numpy as np

import cv2
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.resnet import ResNet 
from models.mlp import MLP

from utils import euclidean_distance, calculate_angle, is_above

class HandSpeak:
    """
    A real-time American Sign Language (ASL) recognition system using hand landmarks.

    This class detects hand gestures using OpenCV and MediaPipe, classifies them into broad 
    groups using a ResNet model, and then predicts specific letters using multiple MLP models. 
    The detected letters are combined into words, which are displayed on a GUI built with Tkinter.

    Args:
        model_group (ResNet): A ResNet model for classifying hand gestures into broad groups.
        model_amnste (MLP): MLP model for predicting specific letters within the 'AMNSTE' group.
        model_dfbuvlkrw (MLP): MLP model for predicting letters within the 'DFBUVLKRW' group.
        model_copqzx (MLP): MLP model for predicting letters within the 'COPQZX' group.
        model_ghyji (MLP): MLP model for predicting letters within the 'GHYJI' group.
        device (str): The computing device ('cpu' or 'cuda') on which models will run.
        group_labels (dict[int, str]): Mapping of ResNet output indices to group names.
        transform (transforms.Compose): Preprocessing transformations applied to input images.
        buffer_size (int, optional): Maximum size for the frame buffer to maintain stability. Defaults to 20.
        typing_delay (float, optional): Time interval (seconds) before appending a new letter. Defaults to 1.3.
    """
    def __init__(self, model_group: ResNet,
                 model_amnste: MLP, model_dfbuvlkrw: MLP, model_copqzx: MLP, model_ghyji: MLP, 
                 device: str, group_labels: dict[int, str], transform: transforms.Compose,
                 buffer_size=20, typing_delay=1.3):
        
        # Given device
        self.device = device
        # ResNet model which predicts the group
        self.model_group = model_group.to(device)
        
        # Load labels.json
        self.encoded_labels = self._load_encoded_labels()
        # Reverse the dictionary to get class label as key and ascii code as value
        self.reversed_encoded_labels = self._reverse_encodes()

        # Save mlp models to predict specific letter
        self.mlp_models = {
            'AMNSTE': model_amnste.to(device),
            'DFBUVLKRW': model_dfbuvlkrw.to(device),
            'GHYJI': model_ghyji.to(device),
            'COPQZX': model_copqzx.to(device),
        }

        self.group_labels = group_labels
        
        # The same transform that is used in train
        self.transform = transform

        # Settings for video capture
        self.capture = cv2.VideoCapture(0)

        # Hand detectors
        self.hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.cropped_hand_detector = HandDetector(detectionCon=0.8, maxHands=1)

        # A small padding for cropping the hand box
        self.PAD = 15

        # Hand landmark connections based on MediaPipe's hand model
        self.hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky Finger
                (5, 9), (9, 13), (13, 17)  # Palm connections
            ]

        # A queue to collect predicted letters
        self.letter_buffer = deque()
        self.prob_threshold = 0.80 # To accept or decline the predicted letter
        self.letter_buffer_size = 7 # Maximum size of letter buffer
        
        # Main frame of HandSpeak
        self.frame = None
        # Frame to display hand landmarks
        self.landmarks_frame = np.ones((350, 350, 3), dtype=np.uint8) * 255

        self.frame_buffer = deque() # Store frames
        self.buffer_size = buffer_size # Maximum size for frame_buffer
        self.typed_text = ""  # Store the final predicted word
        self.last_prediction_time = time.time()  # Timer to handle typing delay
        self.typing_delay = typing_delay # Seconds to wait before adding a new letter

        # Cursor of current letter
        self.cursor_visible = True
        self.cursor_blink_interval = 0.5  # Blink interval in seconds
        self.last_cursor_blink_time = time.time()

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title('American Sign Language')
        self.root.geometry("1000x800")
        self.root.configure(bg='#272727')
        self.root.protocol("WM_DELETE_WINDOW", self._terminate_video_loop)
        self.root.wm_attributes("-topmost", True)
        
        # Remove the default title bar
        self.root.overrideredirect(True)

        # Custom title bar
        self.title_bar = tk.Frame(self.root, bg='#3ba47b', relief='solid', bd=2)
        self.title_bar.pack(fill="x", side="top")

        # Exit button
        self.exit_button = tk.Button(self.title_bar, text="Exit", command=self._close_window, bg='#3ed8c3', fg='white', bd=0)
        self.exit_button.pack(side="right")

        # Clear button to clear typed text
        self.clear_button = tk.Button(self.root, text='Clear', bg='#3ba47b', command=self._clear_text, width=10, height=2)
        self.clear_button.place(x=600, y=470)

        # Make Tkinter window draggable
        self.title_bar.bind("<ButtonPress-1>", self._start_move)
        self.title_bar.bind("<B1-Motion>", self._on_move)

        # A label for main frame (which contains hand gestures)
        self.video_label = tk.Label(self.root, 
                                    bg='black', 
                                    bd=2,
                                    relief='solid',
                                    highlightthickness=3,
                                    highlightbackground="#3ba47b",
                                    highlightcolor="#3ba47b")
        self.video_label.place(x=10, y=35)

        # A label for hand gesture landmarks
        self.landmarks_label = tk.Label(self.root, 
                                    bg='black', 
                                    bd=2,
                                    relief='solid',
                                    highlightthickness=3,
                                    highlightbackground="#3ba47b",
                                    highlightcolor="#3ba47b")
        self.landmarks_label.place(x=600, y=35)

        # Label to display the most common prediction
        self.prediction_label = tk.Label(self.root, text="Prediction: ", bg='#272727', fg='#3ed8c3', font=('Arial', 28, 'bold'))
        self.prediction_label.place(x=10, y=500)

        # Label to display the typed text
        self.typed_text_label = tk.Label(self.root, text="Typed Text: ", bg='#272727', fg='#3ed8c3', font=('Arial', 24, 'bold'))
        self.typed_text_label.place(x=10, y=600)

    def run(self):
        """Starts the Tkinter main loop and begins video processing."""
        self._video_loop()
        self.root.mainloop()

    def _video_loop(self):
        """Captures video frames continuously, processes them, and updates the UI."""
        ret, self.frame = self.capture.read()
        if ret:
            self.frame = cv2.flip(self.frame, 1)  # Flip to simulate mirror image
            self.frame = cv2.resize(self.frame, (500, 500))

            # Re-define the landmarks_frame as white image (3 channels for rgb)
            self.landmarks_frame = np.ones((350, 350, 3), dtype=np.uint8) * 255

            # Detect the hand
            hands, _ = self.hand_detector.findHands(self.frame, draw=False)

            # Predict the letter
            letter = self._predict(hands)

            # Display prediction and text
            display_text = self._text_handler(letter)
            self._tkinter_handler(letter, display_text)

        # Call this function again after 10 milliseconds
        self.root.after(10, self._video_loop)
        
    def _terminate_video_loop(self):
        """Releases the video capture and destroys the Tkinter window."""
        self.capture.release()
        self.root.destroy()

    def _predict(self, hands):
        """
        Runs the model prediction on the given hand landmarks and passes the detected sign group to _predict_letter.

        This method processes detected hand landmarks, applies necessary transformations,
        and feeds the data into a pre-trained model to classify the sign into a broad group.
        The predictions are stored in a buffer to stabilize the results, and the most common
        prediction is used for further classification into a specific letter.

        Args:
            hands (list): A list of detected hands from MediaPipe, where each hand contains landmark data.

        Returns:
            str: The predicted letter corresponding to the detected hand sign, 
                or an empty string if no hand or landmarks are found.

        Notes:
            - If no hands are detected, the function returns an empty string.
            - A moving buffer is maintained to improve prediction stability.
            - The most common prediction in the buffer is used for finer classification into a letter.
        """

        if hands:
            landmarks = self._build_landmarks(hands)
            # no landmarks found
            if landmarks is None:
                return ''
        else: # hand is not detected
            return ''
        
        # Apply the same transformations
        processed_tensor = Image.fromarray(self.landmarks_frame)
        processed_tensor = self.transform(processed_tensor).unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            output = self.model_group(processed_tensor)
            # Get the class number
            _, predicted = torch.max(output, 1)
            class_index = predicted.item()
            # Get the group name
            prediction = self.group_labels[class_index]

            # Add prediction to buffer
            self.frame_buffer.append(prediction)
            # Remove the first element from queue if buffer is full
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.popleft()

            # Get the most common prediction in the buffer
            predicted_group = Counter(self.frame_buffer).most_common(1)[0][0]

            # Predict the specific letter
            return self._predict_letter(predicted_group, landmarks)

    def _predict_letter(self, predicted_group, landmarks):
        """
        Predicts a specific letter from the detected hand landmarks and sign group.

        If the detected sign is classified into a broader group, this method uses a 
        Multi-Layer Perceptron (MLP) model to predict the exact letter. The results are 
        filtered based on a probability threshold, and a moving buffer is used to 
        ensure smooth typing.

        Args:
            predicted_group (str): The broad group classification of the hand gesture.
            landmarks (torch.Tensor or None): Tensor containing hand landmark features, 
                                            or None if no landmarks were detected.

        Returns:
            str: The predicted letter if it meets the probability threshold and appears consistently in previous frames,
                or an empty string if the prediction is uncertain.
        
        Notes:
            - If the sign is classified as 'space', it is returned immediately.
            - A probability threshold is applied to ensure stable predictions.
            - A queue buffer is used to smooth letter predictions.
        """
        with torch.no_grad():
            if predicted_group == 'space':
                return predicted_group
            elif landmarks is not None: # if prediction is not Space and landmarks exist
                landmarks = landmarks.to(self.device)
                
                # Predict the letter using selected model (depends on predicted group)
                selected_model  = self.mlp_models[predicted_group]
                letter = selected_model(landmarks)

                # Get probability of predicted class
                probability = torch.max(F.softmax(letter, dim=1), dim=1)[0].item()
                # Get letter class
                letter = torch.max(letter, 1)[1].item()

                if probability > self.prob_threshold:
                    # Append the predicted letter to queue buffer
                    letter_ascii = self.reversed_encoded_labels[predicted_group][letter]
                    self.letter_buffer.append(chr(letter_ascii))

                    if len(self.letter_buffer) == self.letter_buffer_size:
                        # Get unqiue letters
                        final_letter = list(set(self.letter_buffer))
                        # Dequeue the first element from queue
                        self.letter_buffer.popleft()
                        # If there is one unique letter in previous frames
                        if len(final_letter) == 1:
                            return final_letter[0]
                        else:
                            return ''
                    else:
                        return ''
                else:
                    return ''
            else:
                return ''
            
    def _build_landmarks(self, hands):
        """
        Extracts numerical features from hand landmarks and constructs a feature tensor.

        This method extracts important distances and angles from hand landmarks, 
        which serve as features for gesture classification. It also overlays 
        landmark connections and circles for visualization.

        Args:
            hands (list): A list of detected hands from MediaPipe.

        Returns:
            torch.Tensor or None: A tensor of extracted numerical features if landmarks exist,
                                otherwise returns None.
        
        Notes:
            - If no hands are detected, the function returns None.
            - Key distances and angles between landmarks are computed.
            - The landmarks are drawn onto the frame for visualization.
        """
        numeric_list = []
        if hands:
            x, y, w, h = hands[0]['bbox']
            x1 = max(0, x - self.PAD)
            y1 = max(0, y - self.PAD)
            x2 = min(self.frame.shape[1], x + w + self.PAD)
            y2 = min(self.frame.shape[0], y + h + self.PAD)

            # Crop the hand region
            image = self.frame[y1:y2, x1:x2].copy()

            # Re-detect the hand using the hand region 
            crop_hand, _ = self.cropped_hand_detector.findHands(image, draw=True)

            if crop_hand:
                hand = crop_hand[0]
                landmarks = hand['lmList']
                scale_x = ((350-w)//2)-10
                scale_y = ((350-h)//2)-10

                # Draw lines between hand landmarks            
                for connection in self.hand_connections:
                    pt1 = (landmarks[connection[0]][0]+scale_x, landmarks[connection[0]][1]+scale_y)  # First point (x, y)
                    pt2 = (landmarks[connection[1]][0]+scale_x, landmarks[connection[1]][1]+scale_y)  # Second point (x, y)
                    cv2.line(self.landmarks_frame, pt1, pt2, (0, 255, 0), 2)  # Green lines

                # Draw circles on each landmark
                for lm in landmarks:
                    point_x = lm[0] + scale_x
                    point_y = lm[1] + scale_y
                    cv2.circle(self.landmarks_frame, (point_x, point_y), 5, (0, 0, 255), -1)

                # Distance between the wist and fingertips
                wist_thumb = euclidean_distance(landmarks[0], landmarks[4])
                wist_index = euclidean_distance(landmarks[0], landmarks[8])
                wist_middle = euclidean_distance(landmarks[0], landmarks[12])
                wist_ping = euclidean_distance(landmarks[0], landmarks[16])
                wist_pinky = euclidean_distance(landmarks[0], landmarks[20])

                # Distance between special fingertips
                thumb_index = euclidean_distance(landmarks[4], landmarks[8])
                thumb_pinky = euclidean_distance(landmarks[4], landmarks[20])
                thumb_middle = euclidean_distance(landmarks[4], landmarks[12])
                index_middle = euclidean_distance(landmarks[8], landmarks[12])

                # Other numerical features
                index_middle_dip = euclidean_distance(landmarks[7], landmarks[11])
                index_middle_z = euclidean_distance(landmarks[8][2], landmarks[12][2])
                thumb_ping_angle = calculate_angle(landmarks[4], landmarks[0], landmarks[16])
                thumb_index_angle = calculate_angle(landmarks[4], landmarks[0], landmarks[8])
                index_middle_angle = calculate_angle(landmarks[8], landmarks[0], landmarks[12])

                thumb_index_above = int(is_above(landmarks[4][1], landmarks[8][1]))
                
                numeric_list.extend([wist_thumb, wist_index, wist_middle, wist_ping, wist_pinky,
                                    thumb_index, thumb_middle, thumb_pinky, index_middle,
                                    index_middle_dip, index_middle_z, thumb_ping_angle, thumb_index_angle,
                                    index_middle_angle, thumb_index_above])
        if numeric_list:
            return torch.tensor(numeric_list, dtype=torch.float32).view(1, 15)
        return None

    def _text_handler(self, predicted_letter):
        """
        Handles text input based on predictions and manages cursor blinking.
        
        Args:
            predicted_letter (str): The recognized gesture letter.
        
        Returns:
            str: The updated typed text with cursor display.
        """

        current_time = time.time()
        if current_time - self.last_prediction_time > self.typing_delay and predicted_letter is not None:
            if predicted_letter == 'space':
                self.typed_text += ' '
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

    def _tkinter_handler(self, predict, text):
        """
        Updates the Tkinter UI elements with the latest prediction and typed text.
        
        Args:
            predict (str): The predicted letter.
            text (str): The currently typed text.
        """

        # Prepare frame for displaying
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) # Convert to RGB color space
        img = Image.fromarray(self.frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Prepare landmarks frame for displaying
        l_img = Image.fromarray(self.landmarks_frame)
        l_imgtk = ImageTk.PhotoImage(image=l_img)

        # Put frame on the root window
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        
        # Put landmarks frame on the root window
        self.landmarks_label.imgtk = l_imgtk
        self.landmarks_label.config(image=l_imgtk)

        # Put texts on the root window
        self.prediction_label.config(text=f"Prediction: {predict}")
        self.typed_text_label.config(text=f"Typed Text: {text}")

    def _close_window(self):
        """Closes the Tkinter window and stops the application."""
        self.root.quit()

    def _clear_text(self):
        """Clears the typed text."""
        self.typed_text = ''

    def _start_move(self, event):
        """Stores the initial cursor position when the window move action starts."""
        self.x = event.x
        self.y = event.y

    def _on_move(self, event):
        """Moves the application window based on cursor movement."""
        self.root.geometry(f"+{event.x_root - self.x}+{event.y_root - self.y}")

    def _load_encoded_labels(self):
        """
        Loads encoded labels from a JSON file.

        Returns:
            dict: A dictionary containing label encodings for sign groups.
        
        Notes:
            - The labels are stored in a JSON file located at 'src/labels.json'.
            - This method reads and returns the contents as a dictionary.
        """
        with open('src\labels.json', 'r') as file:
            return json.load(file)
        
    def _reverse_encodes(self):
        """
        Reverses the encoded labels to create a mapping from label values to class indices.

        Returns:
            dict: A dictionary where keys are uppercase group names, 
                and values are dictionaries mapping label values to class indices.
        
        Notes:
            - This method converts encoded labels into a reverse lookup dictionary.
            - It ensures the mapping uses uppercase group names for consistency.
        """
        reverse_encodes = {}
        for group, encodes in self.encoded_labels.items():
            reverse_encode = {v: int(k) for k, v in encodes.items()}
            reverse_encodes[group.upper()] = reverse_encode
        return reverse_encodes