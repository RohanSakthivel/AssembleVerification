import cv2
import numpy as np
import os
import pyttsx3
from tkinter import Tk, Label, StringVar, Frame
from PIL import Image, ImageTk

engine = pyttsx3.init()

# Get available voices
voices = engine.getProperty('voices')

# Set the voice (0 for male, 1 for female)
engine.setProperty('voice', voices[1].id)

# Set the rate (words per minute)
rate = engine.getProperty('rate')
engine.setProperty('rate', 150)  # Default is 200

# Set the volume (0.0 to 1.0)
volume = engine.getProperty('volume')
engine.setProperty('volume', 0.9)

# Directory paths for templates
oilcooler_directory = 'D:/frameimage - Copy/oilcooler'
gasket_directory = 'D:/frameimage - Copy/gasket'
casing_directory = 'D:/frameimage - Copy/casing'

# Load template overlay image
template_overlay_path = 'D:/frameimage - Copy/template.png'
template_overlay = cv2.imread(template_overlay_path, cv2.IMREAD_UNCHANGED)
logo = 'logo.png'

# Instruction images and text
instructions = {
    1: ('D:/frameimage - Copy/instruction 1.png', 'First, you need to place an EGR cooler'),
    2: ('D:/frameimage - Copy/instruction 2.png', 'Second, you need to place a gasket'),
    3: ('D:/frameimage - Copy/instruction 3.png', 'Third, you need to place a casing')
}

# Function to load templates from directory
def load_templates(directory):
    templates = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            template_path = os.path.join(directory, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                templates.append(template)
    return templates

# Tkinter application class
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Assembly Verification")

        # Video frame display
        self.video_frame = Label(root)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        # Frame for feedback and messages
        self.feedback_frame = Frame(root)
        self.feedback_frame.grid(row=0, column=1, padx=10, pady=10, sticky='ns')
        
        # Feedback labels
        self.feedback_label = Label(self.feedback_frame, text="Press 'Start' to begin verification", wraplength=300)
        self.feedback_label.pack(pady=10)

        # Verification status labels
        self.status_labels = []
        for i in range(3):  # Assuming 3 steps: oilcooler, gasket, casing
            status_label = Label(self.feedback_frame, text=f"Step {i+1}: ")
            status_label.pack(pady=5)
            self.status_labels.append(status_label)

        # Count of successfully assembled materials
        self.assembled_count = 0
        self.count_label = Label(self.feedback_frame, text=f"Materials Assembled: {self.assembled_count}")
        self.count_label.pack(pady=10)

        # Instruction display
        self.instruction_image_label = Label(self.feedback_frame)
        self.instruction_image_label.pack(pady=5)
        self.instruction_text_label = Label(self.feedback_frame, text="")
        self.instruction_text_label.pack(pady=5)

        # Gasket image display
        self.gasket_image_label = Label(self.feedback_frame)
        self.gasket_image_label.pack(pady=5)

        # Initialize variables to track detection status
        self.oilcooler_detected = False
        self.gasket_detected = False
        self.casing_detected = False

        # Video capture from file
        self.cap = cv2.VideoCapture('WIN_20240704_15_54_49_Pro.mp4')
        self.step = 1

        # Load templates
        self.oilcooler_templates = load_templates(oilcooler_directory)
        self.gasket_templates = load_templates(gasket_directory)
        self.casing_templates = load_templates(casing_directory)

        # Update video frames
        self.update_frame()
        self.start_verification()

    def start_verification(self):
        self.step = 1
        self.oilcooler_detected = False
        self.gasket_detected = False
        self.casing_detected = False
        self.set_feedback("Starting the verification process...")
        self.verify_components()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def verify_components(self):
        ret, frame = self.cap.read()
        if not ret:
            self.set_feedback("End of video")
            return

        if self.step == 1 and not self.oilcooler_detected:
            self.detect_oilcooler(frame)

        elif self.step == 2 and self.oilcooler_detected and not self.gasket_detected:
            self.detect_gasket(frame)

        elif self.step == 3 and self.oilcooler_detected and self.gasket_detected and not self.casing_detected:
            self.detect_casing(frame)

        # Update instruction display based on current step
        self.update_instruction()

        if self.oilcooler_detected and self.gasket_detected and self.casing_detected:
            self.assembled_count += 1  # Increment the count of assembled materials
            self.count_label.config(text=f"Materials Assembled: {self.assembled_count}")
            self.set_feedback("Material is assembled perfectly")
            engine.say("Material is assembled perfectly")
            engine.runAndWait()
            self.reset_verification()  # Reset to step 1 for the next verification cycle

        self.root.after(1000, self.verify_components)  # Repeat the process every second

    def reset_verification(self):
        self.step = 1
        self.oilcooler_detected = False
        self.gasket_detected = False
        self.casing_detected = False
        for label in self.status_labels:
            label.config(text=f"Step {self.status_labels.index(label) + 1}: ")
        self.set_feedback("Starting new verification cycle...")

    def detect_oilcooler(self, frame):
        detected_position = self.match_templates(frame, self.oilcooler_templates)
        if detected_position:
            self.oilcooler_detected = True
            self.step = 2  # Move to the next step
            self.set_feedback("EGR cooler is placed ✓")
            self.update_status_label(0, "✔")
            engine.say("Good job! EGR cooler is placed perfectly")
            engine.runAndWait()

    def detect_gasket(self, frame):
        detected_position = self.match_templates(frame, self.gasket_templates)
        if detected_position:
            self.gasket_detected = True
            self.step = 3  # Move to the next step
            self.set_feedback("Gasket is placed ✓")
            self.update_status_label(1, "✔")
            self.capture_gasket_image(frame, detected_position)
            engine.say("Good job! Gasket is placed perfectly")
            engine.runAndWait()

    def detect_casing(self, frame):
        detected_position = self.match_templates(frame, self.casing_templates)
        if detected_position:
            self.casing_detected = True
            self.set_feedback("Casing is placed ✓")
            self.update_status_label(2, "✔")
            engine.say("Good job! Casing is placed perfectly")
            engine.runAndWait()

    def update_status_label(self, index, status):
        self.status_labels[index].config(text=f"Step {index + 1}: {status}")
        self.status_labels[index].update()

    def match_templates(self, frame, templates):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = -1
        best_match = None
        for template in templates:
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val_temp, _, max_loc = cv2.minMaxLoc(res)
            if max_val_temp > max_val:
                max_val = max_val_temp
                best_match = max_loc

        threshold = 0.93  # Minimum accuracy threshold set to 93%
        if max_val >= threshold:
            return best_match
        return None

    def set_feedback(self, message):
        self.feedback_label.config(text=message)
        self.feedback_label.update()

    def capture_gasket_image(self, frame, position):
        if position is not None:
            x, y = position
            overlay = self.overlay_image(frame, template_overlay, (x, y))

            # Display the captured gasket image with overlay
            overlay = cv2.resize(overlay, (320, 240))
            overlay_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            overlay_image_tk = ImageTk.PhotoImage(image=overlay_image)
            self.gasket_image_label.config(image=overlay_image_tk)
            self.gasket_image_label.image = overlay_image_tk

    def overlay_image(self, frame, overlay, position):
        x, y = position
        h, w = overlay.shape[:2]

        # Ensure overlay dimensions fit within the frame
        if y + h > frame.shape[0]:
            h = frame.shape[0] - y
            overlay = overlay[:h, :]
        if x + w > frame.shape[1]:
            w = frame.shape[1] - x
            overlay = overlay[:, :w]

        alpha_overlay = overlay[:, :, 3] / 255.0  # Alpha channel of the overlay
        alpha_frame = 1.0 - alpha_overlay

        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                      alpha_frame * frame[y:y+h, x:x+w, c])

        return frame

    def update_instruction(self):
        if self.step in instructions:
            instruction_image_path, instruction_text = instructions[self.step]

            # Load instruction image and convert to Tkinter-compatible format
            instruction_image = Image.open(instruction_image_path)
            instruction_image = instruction_image.resize((150, 120), Image.LANCZOS)
            instruction_image_tk = ImageTk.PhotoImage(instruction_image)

            self.instruction_image_label.config(image=instruction_image_tk)
            self.instruction_image_label.image = instruction_image_tk

            # Set instruction text
            self.instruction_text_label.config(text=instruction_text)
            self.instruction_text_label.update()

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
