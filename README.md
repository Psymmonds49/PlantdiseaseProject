Project Overview

This project implements a Plant Disease Detection System using a Swin Transformer model trained on the PlantVillage dataset. The model is deployed as a Streamlit web application, allowing users to upload leaf images for real-time disease classification.

Dataset

Source: PlantVillage Dataset on Kaggle

Description: The dataset contains images of healthy and diseased plant leaves across multiple species. The model is trained to classify 38 different plant diseases.

1. Install the requirements

   ```
   $ pip install torch torchvision timm streamlit numpy pillow matplotlib scikit-learn seaborn
   ```

How to Run the Project

1. Clone the Repository

   ...

   $ git clone <repository_url>
   cd PlantDiseaseDetection
   ...

2. Download the Dataset

   Download the dataset from Kaggle.

   Extract it to the dataset/ directory inside the project folder.

   ```
   $ streamlit run streamlit_app.py
   ```
3. Train the Model (Optional, if not using pre-trained weights)

   To train the model on your own machine:
      ...
      $ python train_model.py
      ...

   This will train the Swin Transformer model and save the trained weights as swin_transformer_model.pth.

4. Run the Streamlit App

   If using the provided trained model, ensure swin_transformer_model.pth is in the project directory, then run:
      ...
      streamlit run streamlit_app.py
      ...
5. Using the Web App

   Open the URL displayed in your terminal (https://humble-palm-tree-5g5gvj5w7677f7596-8501.app.github.dev/).

   Upload a leaf image.

   The model will classify the plant disease and display the result.
