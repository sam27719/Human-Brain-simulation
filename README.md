NeuroSim EEG Backend is a lightweight FastAPI-based EEG signal analysis and machine-learning backend designed for research, experimentation, and rapid prototyping.

The system provides an API for training a baseline EEG classification model from labeled datasets and using the trained model to predict the class of new EEG signals.

FastAPI REST API for EEG model training and prediction
Supports CSV, NPZ, and NPY EEG datasets
Automatic EEG feature extraction
Time-domain statistical features
FFT-based frequency-domain analysis
Average power extraction across standard EEG frequency bands:
Delta
Theta
Alpha
Beta
Gamma
