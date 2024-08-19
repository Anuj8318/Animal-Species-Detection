# Animal-Species-Detection
This is a simple Flask web application that allows users to upload images for animal detection using a pre-trained MobileNetV2 model. The application can also detect animals in images captured via webcam.

static/uploads/: This folder stores the uploaded images temporarily.
templates/: This folder contains the HTML template files.
app.py: The main Flask application script.
Prerequisites
To run this project locally, ensure you have the following installed:

Python 3.x
Flask
OpenCV
TensorFlow
NumPy
You can install the required dependencies by running:

bash
Copy code
pip install -r requirements.txt

Getting Started
Clone the Repository
bash
Copy code
git clone https://github.com/Anuj8318/Animal-Species-Detection.git
cd animal-detection-webapp
Create Folder Structure
The folder structure should look like this:

bash
Copy code
static/uploads/
templates/
Ensure the uploads folder is created under static/ for storing images.

Run the Application
Start the Flask development server:

bash
Copy code
python app.py
The app will be running on http://127.0.0.1:5000/.

How It Works
Homepage (/): Displays the HTML page for uploading images or capturing images via webcam.
Image Detection (/detect): Handles the detection process for uploaded images or webcam snapshots. It returns the predicted label (animal name).
Routes
GET /: Renders the main page for the web app.
POST /detect: Accepts image uploads or webcam snapshots and returns the detected animal's label.
Example Usage
Upload an image via the homepage or capture one via your webcam.
The application will process the image using OpenCV and MobileNetV2.
The detected label (animal name) will be displayed on the screen
