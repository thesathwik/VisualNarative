# Visual Narrative: CNN-LSTM Image Caption Generator

## Project Overview
This project implements an AI system that generates descriptive captions for images using a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The model is based on the 'Show and Tell' architecture and is implemented using TensorFlow and Keras.

## Dependencies
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- NLTK
- tqdm

## Dataset
The project uses the Flickr8K dataset. Due to computational constraints, a subset of 4,000 entries out of the full 40,000 is used for training and evaluation.

## Data Preprocessing
1. Captions are preprocessed by:
   - Removing extra whitespace
   - Adding `<start>` and `<end>` tokens
2. Images are loaded and resized to 180x180 pixels
3. Image features are extracted using a pre-trained InceptionV3 model (with the top layer removed)

## Model Architecture

### CNN Encoder
- Uses a pre-trained InceptionV3 model (excluding the top layer) for feature extraction
- Adds a fully connected layer to transform the extracted features

### RNN Decoder
- Embedding layer to transform input caption sequences
- LSTM layer for sequence prediction
- Dense layer to transform LSTM output into word predictions

## Training Process
- Uses Adam optimizer
- Implements custom loss function using SparseCategoricalCrossentropy
- Utilizes TensorFlow's `@tf.function` for improved performance
- Implements checkpoint saving and restoration

## Evaluation
- Uses BLEU score (BLEU-1 and BLEU-2) for quantitative evaluation
- Provides functionality to generate captions for individual test images

## Key Features
1. **Feature Extraction**: Uses InceptionV3 for image feature extraction
2. **Tokenization**: Implements vocabulary creation and caption tokenization
3. **Data Pipeline**: Utilizes TensorFlow's Dataset API for efficient data handling
4. **Model Definition**: Custom Keras models for both encoder and decoder
5. **Training Loop**: Implements a custom training loop with gradient tape
6. **Checkpointing**: Saves and restores model checkpoints
7. **Inference**: Provides methods for caption generation on new images
8. **Evaluation**: Implements BLEU score calculation for model evaluation

## Technical Details
- Embedding dimension and LSTM units: 512
- Vocabulary size: Minimum of 5000 words or the actual vocabulary size
- Image dimensions: 180x180 pixels
- Batch size: 64
- Training epochs: 5 (configurable)
- Uses teacher forcing during training
- Implements beam search for caption generation (inference)

## Usage
1. Prepare the Flickr8K dataset
2. Run the preprocessing steps to create image feature dictionaries
3. Train the model using the provided training loop
4. Evaluate the model using the BLEU score calculation
5. Generate captions for new images using the trained model

## Future Improvements
- Implement attention mechanism
- Experiment with different pre-trained CNN models
- Increase dataset size and training epochs for better performance
- Implement more advanced decoding strategies like beam search
