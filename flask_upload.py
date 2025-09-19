from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "data")
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# minimal upload form served by Flask (so you can visit http://localhost:5000)
UPLOAD_FORM = """
<!doctype html>
<title>Upload CSV for Dashboard</title>
<h1>Upload CSV (will be saved as data/uploaded.csv)</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file accept=".csv" required>
  <input type=submit value=Upload>
</form>
<p>After upload you'll be redirected to the Streamlit dashboard at <a href="http://localhost:8501" target="_blank">http://localhost:8501</a>.</p>
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template_string(UPLOAD_FORM)

@app.route('/upload', methods=['POST'])
def upload_file():
    # check file in request
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save as uploaded.csv to a known path the Streamlit app will read
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded.csv")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(save_path)
        # Redirect to Streamlit app (assumed running at 8501)
        return redirect("http://localhost:8501", code=302)
    else:
        return "Invalid file type. Only .csv allowed.", 400

if __name__ == '__main__':
    # Run Flask on port 5000 (default)
    app.run(host='0.0.0.0', port=5000, debug=True)
