# deepfakes_detect
Deepfake Video Detection Using a Hybrid CNN-ViT-Temporal Transformer Architecture
________________________________________

Abstract

Deepfake detection has become a critical challenge due to the increasing sophistication of AI-generated videos. In this work, we propose a hybrid model that integrates Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and a Temporal Transformer for robust deepfake detection. Our approach combines frame-level spatial feature extraction with temporal modeling to capture both visual artifacts and motion inconsistencies present in manipulated videos. We further design a three-stage training strategy for stable and efficient fine-tuning. Experiments on a processed dataset demonstrate that our model achieves 94.96% accuracy with balanced precision, recall, and F1-scores across both original and deepfake classes, proving its effectiveness in real-world detection tasks.
________________________________________

1. Introduction
   
Deepfake videos pose significant threats in misinformation, privacy violations, and digital security. Traditional CNN-based models capture frame-level spatial features but fail to model temporal dependencies. Recently, transformer-based architectures have shown promise in video understanding. To leverage the strengths of both paradigms, we propose a hybrid pipeline that combines CNNs, Vision Transformers (ViTs), and a Temporal Transformer, designed to detect subtle inconsistencies in both frame quality and temporal coherence.
________________________________________
2. Methodology
   
2.1 Preprocessing
•	Raw videos were converted into fixed-length frame sequences.

•	Each frame was resized, normalized using ImageNet statistics, and stored as tensors.

•	Short videos were padded by repeating the last frame.

•	Both image and video inputs were supported, ensuring robust dataset coverage.


2.2 Model Architecture
•	CNN backbone (DenseNet-121): Extracts low-level spatial features.
•	ViT backbone (ViT-Base Patch16-224): Captures global attention-based features from each frame.
•	Projection layers: Map CNN and ViT outputs to a common 256-dim space.
•	Feature fusion: Concatenate CNN and ViT embeddings → 512-dim representation.
•	Temporal Transformer:
o	CLS token aggregates information across frames.
o	Positional embeddings maintain frame order.
o	Transformer layers contextualize temporal dependencies.
•	Classifier: Fully connected layer outputs binary decision (original or deepfake).
2.3 Training Strategy
We adopted a 3-stage curriculum for stable optimization:
1.	Stage 1: Train CNN + projection + temporal transformer + classifier (ViT frozen).
2.	Stage 2: Unfreeze last 2 ViT layers and fine-tune jointly.
3.	Stage 3: Fine-tune the entire model at a smaller learning rate
Mixed precision training was applied to reduce VRAM usage.
________________________________________
3. Results
   
•	Classification Report:
o	Accuracy: 94.96%
o	Precision: Original (0.92), Deepfake (0.97)
o	Recall: Original (0.98), Deepfake (0.92)
o	F1-score: ~0.95 for both classes
•	Confusion Matrix:
•	[[184   4]   → True Originals vs False Deepfakes
•	 [ 15 174]]  → False Originals vs True Deepfakes
These results confirm balanced performance, with the model effectively detecting manipulated videos while minimizing false alarms.

