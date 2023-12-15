

# JointAttention
The Neuroengineering project for the master's course at Politecnico di Milano. 

## Objective

The MLP_CNN_Attention model is designed for a complex classification task involving both numerical data and images. It predicts where a person is looking based on their posture and gaze direction in various settings, including interactions with therapists and robots.

## Data Description
- **Dataset Size**: 4009 entries
- **Composition**:
  - **Numerical Data**: 14 columns representing topological positions of joints and experiment flags.
  - **Images**: Size 120x100, featuring patients and therapists. (Includes some images of walls and robots not directly relevant to the task.)
- **Diversity**: Approximately 20 different pairs of therapists and patients, with varying settings.
- **Post-Augmentation Dataset Size**: Increased to 5597 entries.
- **Potential Improvement**: Future work could focus on refining the dataset, possibly by removing irrelevant entries like robot faces.

## Model Architecture
- **MLP Component**: Processes the numerical data with multiple layers, batch normalization, ReLU activation, and dropout.
- **CNN Component with Attention**: Processes images using a pre-trained EfficientNet, with all layers frozen except the last, to leverage its pre-trained capabilities on a limited dataset, and an added attention mechanism.
- **Integration**: Combines the penultimate layers of MLP and CNN components through a few additional layers.

## Training Process
- **Split Ratio**: Latest split - 60% training, 20% test, 20% validation.
- **Epochs**: 50 for MLP, 5 for CNN, and 5 for the combined model.
- **Optimizer**: Adam with a learning rate of 0.001 and L2 regularization.
- **Data Processing**:
  - **Numerical Data**: Normalization of values by computing mean and variance for each column.
  - **Images**: Normalized to have values between 0 and 1.
- **Data Augmentation**:
  - Addressed unbalanced dataset issues, particularly for labels 1, 4, 3, 5.
  - Aimed to have 600 entries for each underrepresented label through color and noise addition to images.
  - Randomly adjusted numerical data to enhance variability.
- **Loss Function**: CrossEntropyLoss for each component.

## Metrics
-- Achieved an accuracy of 56% and precision of 52% on test data, slightly higher on validation and training sets.

## Observations and Concerns
- **Data Quality**: The presence of irrelevant images and data points affects model performance.
- **Data Processing**: Opportunities exist to refine data processing techniques.
- **Model Complexity and Integration**: Adjustments to the model's integration layer and reliance on a pre-trained CNN indicate potential for further experimentation.
- **Label Distribution**: If certain labels dominate the dataset, the model might be biased towards predicting them.

## Conclusion
While the MLP_CNN_Attention model demonstrates potential, its performance is currently limited by data quality and model complexity. Future efforts should concentrate on improving data quality, exploring data augmentation techniques, and experimenting with model architecture.

### Running the Model on New Data

To evaluate the MLP_CNN_Attention model on new data, follow these steps:

1. **Prepare the Data**:
   - Place the following files in the `data/dataset` directory:
     - `data.csv`: Ensure this file contains 15 columns. The last five columns should include four flags (`robot_patient`, `robot_therapist`, `no_robot_patient`, `no_robot_therapist`) and a numerical column (`ID` of the experiment).
     - `images.npy`: This should be a numpy array of images.
     - `labels.csv`: This file should contain the labels for your data.
   
2. **Data Preprocessing**:
   - Run the `preprocessing/prepare_dataset.py` script. This script processes the data and generates pickle files in the appropriate format for the model.

3. **Model Evaluation**:
   - Use the `evaluate_on_new_data` method in the `training/attention_training.py` script. This script will train the model on your dataset.
   - Use the `evaluate_on_new_data` method within the script to apply the pre-computed weights of the model to your new dataset.

By following these steps, the MLP_CNN_Attention model will be able to process and evaluate your new dataset, giving you insights based on the learned patterns from the training process.
