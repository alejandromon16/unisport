from flask import Flask, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from forms import RegisterForm, LoginForm  # Ensure this line is present
import json
import base64
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os

from modules.modules import capture_faces_from_image

# Create and configure the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Configure the LoginManager
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Define the User class
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# User loader function
@login_manager.user_loader
def load_user(user_id):
    if user_id == '1':
        return User(id=1, username='admin')
    return None

# Helper functions
def save_image(image_file):
    filename = secure_filename(image_file.filename)
    filepath = os.path.join('instance', filename)
    image_file.save(filepath)
    return filepath

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def data_uri_to_cv2_img_list(uri_list):
    img_list = []
    for uri in uri_list:
        img_list.append(data_uri_to_cv2_img(uri))
    return img_list

# Route configurations
@app.route("/", methods=['GET', 'POST'])
@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data == 'admin' and form.password.data == 'admin':
            user = User(id=1, username='admin')
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/home")
@login_required
def home():
    return render_template('home.html', title='Home')

@app.route("/register", methods=['GET', 'POST'])
@login_required
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        image_data = request.form['imageData']
        img = data_uri_to_cv2_img(image_data)
        
        name = form.name.data
        team = form.team.data
        career = form.career.data
        
        capture_faces_from_image(name, team, career, [img])
        flash('Jugador registrado y rostro capturado con éxito!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Registrar', form=form)
    
@app.route("/train")
@login_required
def train():
    try:
        train_faces.train_faces()
        flash('Faces trained successfully!', 'success')
    except ValueError as e:
        flash(str(e), 'danger')
    return redirect(url_for('home'))

@app.route("/capture", methods=['POST'])
@login_required
def capture():
    if request.method == 'POST':
        name = request.form.get('person_name')
        team = request.form.get('team')
        career = request.form.get('career')
        image_data = json.loads(request.form.get('imageData'))
        img_list = data_uri_to_cv2_img_list(image_data)
        capture_faces_from_image(name, team, career, img_list)
        flash('Face captured successfully!', 'success')
        return redirect(url_for('home'))
    return render_template('capture.html', action_url=url_for('capture'))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/recognize", methods=['GET', 'POST'])
@login_required
def recognize():
    if request.method == 'POST':
        image_data = request.form['imageData']
        img = data_uri_to_cv2_img(image_data)
        
        recognized_name, recognized_team, recognized_career = recognize_face(img)
        
        if recognized_name:
            message = f"Bienvenido Jugador {recognized_name} de {recognized_team}, de la {recognized_career}"
        else:
            message = "Rostro no reconocido. Inténtalo de nuevo."
        
        return render_template('recognize.html', message=message)
    return render_template('recognize.html')

def recognize_face(img):
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classif.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None, None

    (x, y, w, h) = faces[0]
    rostro = img[y:y+h, x:x+w]
    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
    
    # Load known faces and labels
    with open('instance/players.json', 'r') as f:
        players = json.load(f)
    
    for player_name, player_info in players.items():
        for image_name in player_info['images']:
            known_image_path = os.path.join(player_info['path'], image_name)
            known_image = cv2.imread(known_image_path)
            
            if known_image is None:
                continue
            
            known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
            
            # Compute similarity (using a simple method like mean squared error for demonstration)
            diff = cv2.absdiff(known_gray, cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY))
            mse = np.mean(diff ** 2)
            
            if mse < 1000:  # Threshold value, may need adjustment
                return player_name, player_info['team'], player_info['career']
    
    return None, None, None



# Run the app
if __name__ == "__main__":
    app.run(debug=True)
