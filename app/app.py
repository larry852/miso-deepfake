from flask import Flask
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import script 

UPLOAD_FOLDER = 'static/videos/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        images = script.run(filename)
        prediction = images['Prediction'].value_counts().idxmax()
        flash('Video analizado exitosamente')
        return render_template('upload.html', filename=filename, images=images.to_dict("records"), prediction=prediction)

@app.route('/display-video/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='videos/' + filename), code=301)

@app.route('/display-image/<video>/<filename>')
def display_image(video, filename):    
    return redirect(url_for('static', filename='videos/' + video.replace('.mp4','') + '/faces/' + filename))
