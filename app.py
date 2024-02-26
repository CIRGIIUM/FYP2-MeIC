from flask import Flask, render_template, request, jsonify
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

app = Flask(__name__)


#map datasets to model paths
model_paths = {
    'cataract': 'models/cataract.h5',
    'leukemia': 'models/leukemia.h5',
    'pneumonia': 'models/pneumonia.h5',
    'skincancer': 'models/skincancer.h5'
}

#class labels
class_labels = {
    'cataract': ['Normal', 'Cataract'],  
    'leukemia': ['Normal', 'Leukemia'], 
    'pneumonia': ['Normal', 'Pneumonia'],
    'skincancer': ['Skincancer(Benign)', 'Skincancer(Malignant)', 'Skincancer(Vascular)', 'Skincancer(Dermatofibroma)'] 
}

@app.route('/MeIC', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    dataset = request.form['dataset']
    
    if dataset not in model_paths:
        return jsonify({'error': 'Invalid dataset selection'})
    
    model_path = model_paths[dataset]
    class_label = class_labels[dataset]
    
    model = load_model(model_path)

    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
   
    yhat = model.predict(image)
    
  
    predicted_index = int(yhat.argmax(axis=-1))


    predicted_class = class_label[predicted_index]
    
    classification = f'{predicted_class}'

    return jsonify({'prediction': classification})

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
