from flask import Flask, render_template, request, redirect, send_file
from data_preprocessing import preprocess_data
from model import train_model, predict_moisture

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    file.save('static/uploaded_data/' + file.filename)
    return redirect('/predict/' + file.filename)

@app.route('/predict/<filename>')
def predict(filename):
    data = preprocess_data('static/uploaded_data/' + filename)
    predicted_data = predict_moisture(data)
    return render_template('index.html', predicted_data=predicted_data)

@app.route('/download/<filename>')
def download(filename):
    return send_file('static/uploaded_data/' + filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
