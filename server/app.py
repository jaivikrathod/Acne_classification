from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from filter5 import main

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Create 'uploads' directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file to a temporary location (optional)
    file_path = 'uploads/' + file.filename
    file.save(file_path)
    
    # Call the main function from filter5.py
    try:
        # Assuming 'main' function expects the file path
        result = main(file_path)  
        # 0-9 [1]  10-15[2] 16-25[3] 26-32[4] 32< [5]  
        # result ={
        #     'Detected_acne_marks' : "0",
        #     'severety_level' :0,
        #     'suggested medicine' :['none']
        # }
            
        # return jsonify({"message": "No acne detected", "result": result}), 200
        # return jsonify({"message": "No Face Detected", "result": result,"isFace":1}), 200
        return jsonify({"message": "Multiple faces Detected", "result": result,"isFace":2}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
