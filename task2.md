Shoplifting Detection 
Objective:
Build a video classification model that distinguishes between normal shopping
behavior and shoplifting/theft acts, by training on labeled video clips using
temporal deep learning architectures.
Tasks
Search Hugging Face Hub for pretrained video classification models (e.g., VideoMAE,
TimeSformer, or X3D).
Load the chosen pretrained model and adapt its classifier head for binary classification
(normal vs shoplifting).
Reuse the frame sampling and augmentation pipeline from Week 5. • Fine-tune the
pretrained model on the shoplifting dataset.
Compare pretrained model performance against the from-scratch model from Week 5.
Discuss the trade-offs in inference speed, model size, and accuracy.
Completion Requirements
A pretrained video model is loaded from Hugging Face without errors.
The classifier head is correctly adapted for 2-class output.
Fine-tuned model achieves higher F1-score than the Week 5 scratch model.
A comparison table of both models is included
Code is pushed to GitHub
Resources
https://huggingface.co/models?pipeline_tag=video-classification