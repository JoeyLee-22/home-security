import cv2
import face_recognition as face_rec
import numpy as np

class liveFaceRec:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        
        self.known_face_encodings = []
        self.known_face_names =[]
        
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        self.process_this_frame = True
    
    def get_sample(self):
        # Load a sample picture and learn how to recognize it.
        image_1 = face_rec.load_image_file("photos/joey.jpeg")
        image_1_encoding = face_rec.face_encodings(image_1)[0]

        image_2 = face_rec.load_image_file("photos/jasper.jpeg")
        image_2_encoding = face_rec.face_encodings(image_2)[0]

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            image_1_encoding,
            image_2_encoding,
        ]
        self.known_face_names = [
            "Joey Lee",
            "Jasper Lee",
        ]

    def live_rec(self):
        while True:
            # Grab a single frame of video
            ret, frame = self.video_capture.read()

            # Only process every other frame of video to save time
            if self.process_this_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_rec uses)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_rec.face_locations(rgb_small_frame)
                self.face_encodings = face_rec.face_encodings(rgb_small_frame, self.face_locations)

                face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_rec.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_rec.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)
            self.process_this_frame = not self.process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        self.video_capture.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    lfr = liveFaceRec()
    lfr.get_sample()
    lfr.live_rec()