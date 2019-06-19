import face_recognition
import os
import statistics

encodings = []
train_dir = os.path.abspath('photos')
test_image = "kevin.jpg"

# Loop through each person in the training directory
# Loop through each training image for the current person
for photos in os.listdir(train_dir):
    # Get the face encodings for the face in each image file
    photo = face_recognition.load_image_file(train_dir + '/' + photos )
    photo_encoded = face_recognition.face_encodings(photo)[0]
    # Add face encoding for current image with corresponding label (name) to the training data
    encodings.append(photo_encoded)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file(test_image)

# # Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
face_distances = face_recognition.face_distance(encodings, face_recognition.face_encodings(test_image)[0])

for image_number, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, image_number))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()

print(f"Similarity: {100 - statistics.mean(face_distances * 100):.2f}%")
