# Acoustic Footstep Identification and Localization Model

This repository contains a Convolutional Recurrent Neural Network (CRNN) model for acoustic footstep identification and localization. The model processes audio input from multiple microphones to:
1. Identify the subject (person) based on their footstep sound
2. Localize the position of the footstep in 2D space (x, y coordinates)

## Model Architecture

The model uses a hybrid architecture combining:
- Convolutional layers for processing GCC-PHAT and Mel-spectrogram features
- GRU (Gated Recurrent Unit) layer for temporal processing
- Dense layers for final prediction

The model takes two types of input features:
1. GCC-PHAT (Generalized Cross-Correlation with Phase Transform) features
2. Mel-spectrogram features

## Usage

```python
from model import build_crnn_model, extract_features

# Extract features from audio file
gcc_features, spec_features = extract_features('path_to_audio.wav')

# Build and compile the model
model = build_crnn_model(
    input_shape_gcc=(gcc_features.shape[1], gcc_features.shape[2]),
    input_shape_spec=(spec_features.shape[1], spec_features.shape[2]),
    hidden_units=128,
    output_units=10,  # Number of subjects to identify
    dropout_rate=0.2
)

# Compile the model
model.compile(
    optimizer='adam',
    loss={
        'output_loc_x': 'mse',
        'output_loc_y': 'mse',
        'output_subject': 'categorical_crossentropy'
    },
    metrics={
        'output_loc_x': 'mae',
        'output_loc_y': 'mae',
        'output_subject': 'accuracy'
    }
)

# Make predictions
x_pred, y_pred, subject_pred = model.predict([gcc_features, spec_features])
```

## Input Requirements

The audio input should be:
- Multi-channel (4 channels)
- Sampling rate: 16000 Hz
- Format: WAV file

## Output Format

The model outputs:
1. x-coordinate prediction (output_loc_x)
2. y-coordinate prediction (output_loc_y)
3. Subject identification probabilities (output_subject) 
