# Training Report

## Model
- U-Net for image segmentation 
- Encoder: ResNet-34 (pretrained on ImageNet)
- Decoder: U-Net with skip connections
- Input size: 512 × 512
- Output: Binary mask (calculator vs background)

## Training Details
- Epochs: 50
- Batch size: 4
- Optimizer: Adam
- Learning rate: 1e-4
- LR scheduler: ReduceLROnPlateau
- Loss function: BCE + Dice loss
- Early stoping used just to overcome overfiitng

## Dataset
- Training: 80 images
- Validation: 21 images
- Test: 11 images

## Data Augmentation
- Used data augmentation (training dataset only)

## Results (Test Set)
- Loss: 0.2651
- IoU: 0.8728
- Precision: 0.9561
- Recall: 0.8998
- F1 Score: 0.9271

## Observations
- Model trains smoothly without errors
- Good precision, fewer false positives
- Slightly lower recall than precision
- Small validation set makes results noisy
- Some overfitting due to small dataset but ignorable

## Limitations
- Dataset is very small 
- Validation set is aslo small as compared to training set
- Model may not work well in hard cases 

## My Conclusion
I used a U-Net model for image segmentation because it works well with small datasets and trains fast. The model performed well on the test set and gave good accuracy with strong IoU and precision scores. Overall, the results are good, but the small dataset is a main limitation. The model can become better in the future if more training data is added more.

## Visual Results
- Training curves: `results/training_curves.png`
- Test predictions: `results/test_predictions.png`