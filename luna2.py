import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
from pathlib import Path
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv('AIzaSyCggv7ZZsDFNiyUHYmY9ZM4g4eZ-yR1g_Q')
genai.configure(api_key=GOOGLE_API_KEY)

class AIVisionChat:
    def __init__(self):
        # Initialize YOLO model
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        
        # Initialize Gemini model
        print("Connecting to Gemini API...")
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera!")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Keywords for vision-related questions
        self.vision_keywords = ['what is that', 'what do you see', 'tell me what', 'identify', 'show me', 'describe']
        
        print("System ready!")
    
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLOv8."""
        results = self.model(frame, conf=0.5)
        return results[0]
    
    def process_voice_input(self):
        """Listen for voice input and convert to text."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError:
                print("Could not request results")
                return None
    
    def speak_response(self, text):
        """Convert text to speech."""
        print(f"AI: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
    
    def get_ai_response(self, question, detected_objects=None):
        """Get response from Gemini for both general questions and object detection."""
        try:
            if detected_objects:
                # Vision-related question
                objects_desc = ", ".join([f"{obj['name']} (confidence: {obj['conf']:.2%})" 
                                        for obj in detected_objects])
                prompt = f"""I can see the following objects: {objects_desc}. 
                        The question asked was: '{question}'
                        Please provide a natural, conversational response about what you see,
                        focusing on the most prominent object if there are multiple."""
            else:
                # General question
                prompt = f"""Please provide a clear and concise answer to this question: {question}
                        Keep the response natural and conversational."""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            if detected_objects:
                return f"I can see {objects_desc}, but I'm having trouble forming a detailed response right now."
            else:
                return "I apologize, but I'm having trouble generating a response right now. Please try asking your question again."
    
    def is_vision_question(self, question):
        """Determine if the question is related to visual detection."""
        return any(keyword in question for keyword in self.vision_keywords)
    
    def draw_detections(self, frame, results):
        """Draw detection boxes and labels on the frame."""
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            cls_id = box.cls[0].item()
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f"{results.names[cls_id]} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop for the application."""
        print("Starting AI Vision Chat. Say 'quit' to exit.")
        print("You can ask any general questions or ask about objects you show!")
        
        last_detection_time = 0
        detection_cooldown = 2
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Always run detection for visual feedback
            results = self.detect_objects(frame)
            
            # Draw detection boxes
            frame_with_boxes = self.draw_detections(frame, results)
            
            # Show frame
            cv2.imshow('Camera Feed (Press Q to quit)', frame_with_boxes)
            
            # Process voice input
            voice_input = self.process_voice_input()
            
            current_time = time.time()
            if voice_input:
                if voice_input == 'quit':
                    break
                
                if current_time - last_detection_time > detection_cooldown:
                    last_detection_time = current_time
                    
                    if self.is_vision_question(voice_input):
                        # Handle vision-related questions
                        detected_objects = []
                        for box in results.boxes:
                            cls_id = box.cls[0].item()
                            conf = box.conf[0].item()
                            detected_objects.append({
                                'name': results.names[cls_id],
                                'conf': conf
                            })
                        
                        if detected_objects:
                            response = self.get_ai_response(voice_input, detected_objects)
                        else:
                            response = "I don't see any objects clearly right now. Could you please adjust the object or lighting?"
                    else:
                        # Handle general questions
                        response = self.get_ai_response(voice_input)
                    
                    self.speak_response(response)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        app = AIVisionChat()
        app.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()