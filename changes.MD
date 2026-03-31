Context: In my project I attempt to replicate the FaceNet nns2 architecture from Googles FaceNet paper in order to build a Google Photos replica.
It would take small faces from photos and then run them through the FaceNet which would use a triplet loss function in order to generate a 128-dimensional vector of facial detections. 

Problems: It has very low ARI score and the accuracy is not good, several problems with the performance of the model.

Task: look at what my code analysis tool wrote as a response for possible improvement, read my suggestions and modify the code.

FaceNet Implementation Analysis Report
Critical Issues
1. Inadequate Network Architecture
The TinyFaceNet is severely underpowered compared to the original FaceNet paper. The original uses inception modules with 140M parameters; yours has ~2M parameters with only 3 conv layers.

Impact: The model cannot learn complex facial features needed for reliable discrimination.

Fix: Implement proper inception blocks or use ResNet-50 backbone.