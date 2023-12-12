# JointAttention
The Neuroengineering project for the master's course at Politecnico di Milano. 

# Project Overview: MLP_CNN_Attention Model

## Objective

The MLP_CNN_Attention model is designed for a complex classification task involving both numerical data and images. It predicts where a person is looking based on their posture and gaze direction in various settings, including interactions with therapists and robots.

## Data Description
- **Dataset Size**: 4009 entries
- **Composition**:
  - **Numerical Data**: 14 columns representing topological positions of joints and experiment flags.
  - **Images**: Size 120x100, featuring patients and therapists.
- **Diversity**: Approximately 20 different pairs of therapists and patients, with varying settings.

## Model Architecture
- **MLP Component**: Processes the numerical data with multiple layers, batch normalization, ReLU activation, and dropout.
- **CNN Component with Attention**: Processes images using a pre-trained EfficientNet and an added attention mechanism.
- **Integration**: Combines outputs of MLP and CNN components through a final linear layer.

## Training Process
- **Split Ratio**: Latest split - 60% training, 20% test, 20% validation.
- **Epochs**: 50 for MLP, 5 for CNN, and 5 for the combined model.
- **Optimizer**: Adam with a learning rate of 0.001 and L2 regularization.
- **Loss Function**: CrossEntropyLoss for each component.

## Observations and Concerns
- **Data Overfitting**: The model might be too complex for the data size, learning to memorize rather than generalize. However, it is hard to judge as it also performs well on unseen data.
- **Data Homogeneity**: Lack of variability in data might make it easy for the model to predict.
- **Small Dataset Size**: A larger dataset might present more challenges and diverse scenarios.
- **Pre-trained CNN Influence**: The use of a powerful pre-trained network may contribute to high performance on a potentially simple dataset.
- **Evaluation Methodology**: The way the validation and test sets are used or the metric calculations might be inflating performance figures.
- **Label Distribution**: If certain labels dominate the dataset, the model might be biased towards predicting them.

## Future Directions
- **Data Expansion**: More data in varied locations with a diverse range of individuals.
- **Model Complexity**: Assess and potentially simplify the model to match dataset complexity.
- **Data Augmentation**: Implement techniques to increase dataset variability.
- **Cross-Validation**: Ensure model generalizability using cross-validation.
- **Experimentation**: Test different architectures, including simpler models.

## Conclusion
The MLP_CNN_Attention model shows promise in its task but raises questions about its high performance metrics. Future efforts should focus on dataset expansion, model adjustment, and rigorous validation techniques to ensure reliability and generalizability in real-world scenarios.
