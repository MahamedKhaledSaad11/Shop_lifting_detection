Shoplifting Detection 
Objective:
Build a video classification model that distinguishes between normal shopping
behavior and shoplifting/theft acts, by training on labeled video clips using
temporal deep learning architectures
Tasks
Explore the video dataset: number of clips per class (normal / shoplifting), clip duration
distribution, and frame resolution.
Implement a frame sampling strategy: given the total frame count of each video and a
target frame count T, compute a uniform skip interval so that sampled frames span the
full video duration as evenly as possible.
Apply consistent spatial augmentation across all sampled frames of the same video
(same random flip, crop, and color jitter must be applied to every frame in the clip).
Build a PyTorch Dataset class that returns a tensor of shape (T, C, H, W) per video.
Implement one of the following architectures: (a) 3D CNN operating on the full clip
tensor, or (b) CNN feature extractor (e.g., ResNet) per frame followed by an RNN/GRU for
temporal aggregation.
Train the model and evaluate with accuracy, precision, recall, and F1-score.
Completion Requirements
Frame sampling code correctly samples T frames from any video regardless of
its length, verified on clips of different durations
Augmentation is applied identically to all frames within the same video clip
(verified visually or via code check
Dataset class returns tensors of consistent shape (T, C, H, W).
At least one temporal architecture (3D CNN or CNN+RNN) is implemented and
trains without errors
Evaluation metrics (accuracy, F1) are reported on the test set. • Code is pushed
to GitHub