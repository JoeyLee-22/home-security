from deepface import DeepFace

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

# face verification
try:
    verification = DeepFace.verify(img1_path = "template.jpeg", img2_path = "tocheck.jpeg")
    print("\nFace Verified: " + (str)(verification["verified"]) + "\n")
except ValueError:
    print("\nNo face detected in the image\n")

# analyze image
try:
    analysis = DeepFace.analyze(img_path = "template.jpeg", actions = ["age", "gender", "emotion", "race"])
    print("\nAge:     " + (str)(analysis[0]["age"]))
    gender = analysis[0]["gender"]
    print("Gender:  " + (str)(max(gender, key=gender.get)))
    print("Emotion: " + (str)(analysis[0]["dominant_emotion"]))
    print("Race:    " + (str)(analysis[0]["dominant_race"]))
except ValueError:
    print("\nNo face detected in the image")

# realtime face recognition
DeepFace.stream(db_path = "me", model_name = models[2])