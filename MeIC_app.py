from flask import Flask, render_template, request, jsonify, redirect
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
# import tensorflow_privacy

app = Flask(__name__)



#map datasets to model paths
model_paths = {
    'cataract': 'models/dp_cataract.h5',
    'leukemia': 'models/dp_leukemia.h5',
    'pneumonia': 'models/dp_pneumonia.h5',
    'skincancer': 'models/dp_skincancer.h5'
}

#class labels
class_labels = {
    'cataract': ['Normal', 'Cataract'],  
    'leukemia': ['Normal', 'Leukemia'], 
    'pneumonia': ['Normal', 'Pneumonia'],
    'skincancer': ['Skincancer(Benign)', 'Skincancer(Malignant)', 'Skincancer(Vascular)', 'Skincancer(Dermatofibroma)'] 
}

@app.route('/')
def redirect_to_meic():
    return redirect('/MeIC')

@app.route('/MeIC', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        imagefile = request.files['imagefile']
        dataset = request.form['dataset']

        if dataset not in model_paths:
            return jsonify({'error': 'Invalid dataset selection'})

        model_path = model_paths[dataset]
        class_label = class_labels[dataset]

        print("Model Path:", model_path)  # Debug statement to print model path

        model = load_model(model_path)

        print("Model Loaded Successfully!")  # Debug statement to confirm model loaded successfully

        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        # Resize image to match model input shape
        target_size = (128, 128)  # Adjust dimensions as needed
        image = Image.open(image_path)
        image = image.resize(target_size)
        image = img_to_array(image)

        # Expand dimensions to match model input shape
        image = np.expand_dims(image, axis=0)

        image = preprocess_input(image)

        yhat = model.predict(image)

        predicted_index = int(yhat.argmax(axis=-1))
        predicted_class = class_label[predicted_index]
        classification = f'{predicted_class}'

        return jsonify({'prediction': classification})
    except Exception as e:
        # Print the exception traceback to help identify the issue
        import traceback
        print("Exception occurred:", e)
        traceback.print_exc()

        return jsonify({'error': 'An error occurred during prediction. Please try again later.'})

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(port=5003, debug=True)
