# Project Status

## Current state

- Local VS Code project created
- Virtual environment created successfully
- Python version: 3.12.6
- Terminal could not detect NVIDIA GPU using nvidia-smi
- Next step: move old Colab notebooks and inspect pipeline

## Old thesis work

- Best previous model: InceptionV4
- Within-dataset accuracy: about 75.27%
- Cross-dataset accuracy: 64.8%
- Professor requested:
  - error analysis
  - dataset comparison
  - label quality checking
  - preprocessing/cropping consistency check
  - confirmation about face detection

- Clean local Python modules created:
  - paths.py
  - transforms.py
  - model.py
  - dataset.py
- Import test passed successfully
- Dataset loader module created

- Clean local training on Dataset 1 completed
- Epoch 1: train loss 1.2984, val acc 0.5312
- Epoch 2: train loss 0.8185, val acc 0.6737
- Final Dataset 1 test accuracy: 0.6570
- Model saved as dataset1_inception_v4_baseline.pth

## Cross-Dataset Evaluation (Dataset1 → Dataset2)

- Accuracy: 0.6000
- Weighted F1-score: 0.6050

### Observations:
- Performance dropped compared to Dataset 1 test accuracy (~0.6860)
- Some classes are easier:
  - square: strong performance
  - heart: relatively good
- Some classes are difficult:
  - oval: weakest performance (confused with multiple classes)
  - round vs oval confusion

### Confusion insights:
- oval often misclassified as:
  - heart
  - round
  - square
- oblong confused with oval
- round confused with oval

### Conclusion:
- Face shape classification is sensitive to dataset differences
- Similar face shapes (oval, oblong) are difficult to distinguish
