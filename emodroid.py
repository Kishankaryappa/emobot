import pyttsx3
import cv2
import numpy as np
from keras.models import model_from_json
from collections import deque

engine = pyttsx3.init()

def recognize_text():
    return input("You: ")

def speak(text):
    print("Emo:", text)
    engine.say(text)
    engine.runAndWait()

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")
speak("Fetching for the Pre-Trained Model........Please wait ")
print("Loading The Model........Please wait ")

emotion_buffer = deque(maxlen=5)
min_unique_emotions = 2

def open_camera():
    speak("Let me open your device's camera with face tracking and emotion recognition. Please wait a moment , Keep the camera in front of your Face.")
    cap = cv2.VideoCapture(0)

    emotion_counter = 0
    emotions = []

    while emotion_counter < 15:
        ret, frame = cap.read()
        if not ret:
            break

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(num_faces) == 0:
            continue

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(emotion_prediction))
            detected_emotion = emotion_dict[max_index]
            emotions.append(detected_emotion)

            cv2.putText(frame, detected_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            print("Detected emotion:", detected_emotion)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        emotion_counter += 1

    cap.release()
    cv2.destroyAllWindows()

    if emotions:
        average_emotion = max(set(emotions), key=emotions.count)
        speak(f"It appears you are {average_emotion} right now.")
        interact(average_emotion)


def greet():
    greetings = [
        "Welcome to the world of EmoDroid, your friendly chatbot companion! What's the name that brings a smile to your face?",
        "Hey there! I'm EmoDroid, your personal AI sidekick. Mind sharing your awesome name with me?",
        "Greetings, esteemed human! I am EmoDroid, the chatbot designed to make your day brighter. May I have the honor of knowing your delightful name?",
        "Ahoy there! EmoDroid at your service, ready to assist you with any task. What shall I call you, dear traveler?",
        "Salutations, kind soul! I'm EmoDroid, here to accompany you on your virtual adventures. Would you be so kind as to share your name with me?"
    ]

    for greeting in greetings:
        speak(greeting)
        name = recognize_text()
        if name:
            speak(f"Hello, {name}! I'm glad to be your assistant. How can I assist you today?")
            open_camera()
            break

def goodbye():
    speak("Goodbye! Have a great day!")

def interact(emotion):
    if emotion == "Happy":
        speak("I'm glad to hear that you're feeling happy!")
        speak("Always keep smiling , its healthy!!")
    elif emotion == "Sad":
        speak("I'm sorry to hear that you're feeling sad.")
        speak("Past is past , move on my friend. go out and explore the world")
    elif emotion == "Surprise":
        speak("It seems you are surprised to see my abilities")
    elif emotion == "neutral":
        speak("Take chill pill , ill suggest you some songs for improve your mood")
    elif emotion == "Angry":
        speak("be calm , take a deep breath")
        
    else:
        speak("It's interesting to know your current emotion!")
        

    speak("How are you feeling now?")
    mood = recognize_text()
    if mood:
        speak(f"Thank you for sharing your mood!")
    else:
        speak("Sorry, I didn't understand. Could you please rephrase?")
    speak("Do you want to continue? (yes/no)")
    response = recognize_text()
    if response and response.lower() == "yes":
        open_camera()
    else:
        goodbye()

def chat():
    greet()
    while True:
        speak("Do you want to continue? (yes/no)")
        response = recognize_text()
        if response and response.lower() == "no":
            speak("Thanks for your time , I tried my best to give 70 to 80 percent accurate recognition , and i improve with time ")
            goodbye()
           
            break
        elif response and response.lower() == "yes":
            open_camera()
        else:
            speak("Sorry, I didn't understand. Could you please rephrase?")

chat()