#  Setup steps
# 1. Install dependencies
pip install streamlit streamlit-drawable-canvas tensorflow numpy opencv-python

# 2. Train the CNN weights
python train_model.py using mnist_cnn_enhanced.h5

# 3. Run Streamlit Dashboard App
streamlit run app.py

# 4. App Explanation
Your Canvas Drawing (280x280 Large Matrix)
          │
          ▼
OpenCV Bounding Box (cv2.findContour |  cv2.boundingRect) ──► Locates bounding limits of white brush strokes and Strips away empty space and handles corners
          │
          ▼
Centering & Padding (cv2.copyMakeBorder) ──►  Pads the cropped shape to avoid squishing
          │
          ▼
Resize & Blur (cv2.resize) ──► Compresses smoothly into standard 28x28 neural tensor
          │
          ▼
CNN Model Pass [model.predict(input_tensor, verbose=0)] ──► This method sends given 28x28 normalized canvas matrix through the layers of the trained network to compute the raw classification vector.
    -> Load and Normalize Dataset
    -> Build the CNN Architecture
    -> First Convolutional Block :- 
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)) -: Creates 32 unique tiny visual filters. 
    layers.BatchNormalization(): Standardizes the outputs coming out of the filters.
    layers.Activation('relu'): Applies the Rectified Linear Unit math equation.
    layers.MaxPooling2D((2, 2)): Divides the image into tiny 2x2 grids and extracts only the brightest, most dominant pixel value from each grid.
    layers.Dropout(0.25): Randomly breaks/deactivates 25% of the neural connections in this block during training. 
    -> Second Convolutional Block: Works identically to the first block, but bumps the filter count up to 64. Because the spatial map has shrunk, these filters combine the basic lines found in block one to look for more complex multi-stroke features like loops, closed circles, and intersections.MaxPooling & Dropout will shrinks the spatial data map down and applies another 25% dropout guardrail.
    -> layers.Flatten()
    -> layers.Dense(256): A fully-connected layer containing 256 regular brain-like neurons works to figure out how they connect to human numbers.
    -> layers.Dropout(0.5): A heavier 50% dropout safety buffer applied right before the final score calculation. 
    -> layers.Dense(10, activation='softmax'): This is final step, It has exactly 10 nodes (one for each digit from 0 to 9). The Softmax formula squashes the internal numeric signals into a probability list where all ten output options add up to exactly 1.0 or 100%.
          │
          ▼
Softmax Layer [layers.Dense(10, activation='softmax') and np.argmax()] ──► This model's final dense layer uses the softmax activation to output 10 fractions. np.argmax() looks at that array and extracts the index of the highest probability value, which becomes your predicted_digit.
          │
          ▼
Streamlit UI [st.columns() and st.metric()] ──► Streamlit UI methods take the processed predicted_digit and confidence_score floats and render them inside clean, UI layout components.
          │
          ▼
Plotly Charting Engine [go.Figure(), go.Bar(), and st.plotly_chart()] ──► Visualizes the model's top 3 confidence probabilities live on Plotly Chart Engine. 
About Functions:- The go.Bar function maps your prediction_probabilities array into vertical columns, highlights the predicted one in orange-red, saves it inside fig, and uses st.plotly_chart to render it onto your webpage layout.


# Test Results
1. Standard Centered - Drawn cleanly in middle
2. Edge/Corner Bias - Drawn small in far-right corner
3. Thin Stroke - Mouse clicked/dragged swiftly 
4. Thick Stroke - Heavy brush track
5. Structural Ambiguity - Messy, unclosed loop (e.g. 0 vs 6)

# Reflection Answers
How does a computer represent a handwritten digit as data?
Computer represent handwritten digits as 2D pixel matrixs. Here in this project we have standard 28×28 grayscale image is translated into data We are using Streamlit application for UI, TensorFlow, Keras CNN model named mnist dataset enhanced h5 model is later trained to identify handwritten value.

Why does the model use a 28 × 28 image?
we have preset the image dimensions specifically 28 x 28 = 784 features are small dimensions gives us more accurate input to allow early neural networks and support vector machines execution, yet large enough to capture the complex curves and loops of handwritten digits.

Why are pixel values normalized between 0 and 1?
It is done for convenience of computer as pixel values between 0,1 is easiler to process. Large values (0,255) do weight updates in a neural network, causing training to fail or gets really slow. By normalizing the data, you convert the computer's raw representation into floating-point numbers between 0.0 (black background) and 1.0 (pure white stroke).

What does the confidence score represent?
It represents the probability that your drawn digit belongs to a specific class between (0-9). Its last layer returns 10 raw values. The softmax activation takes those raw scores and squashes them into a normalized range between 0 and 1.

Which type of drawing was hardest for your model to classify, and why do you think that happened?
In classic MNIST digit classification, 9 and 5 are the hardest to classify due to overlapping geometry and messy handwriting. This also happens because computers lack human intuition and rely entirely on spatial math.

What's the difference between training a model and using a trained model for inference?
The basic difference is a model training is the process of a model "learning" a skill, while inference is using that learned skill to make a prediction on new data.
Training a model - # The model is actively learning here.
model.fit(train_images, train_labels, epochs=10)
Predicting a model (Inference) - # The model is frozen; it is just using its brain to guess.
prediction = model.predict(demo_img)

How could you improve this application in the future?
Next step is make it production ready by deploying this streamlit drawable canvas application. After that we can share it on Hugging-Face or Streamlit-Community Space. Also can solve the "Real-World Data Drift" Problems (Accept real-user usuablity challanges). Add Horizontal chart of top 3 predictions, Add a Freedback button under predictions to know if predicted result is right or wrong!