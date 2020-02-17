# RecurrentBIM-PoseNet
This is an implementation of Recurrent BIM-PoseNet for camera pose regression related to our upcoming work. More details will be available soon.  

The training and the test data can be found in [this](https://melbourne.figshare.com/articles/UnimelbCorridorSynthetic_zip/10930457) repository. 

## Dataset preview
[![Watch the video](https://melbourne.figshare.com/ndownloader/files/19441991/preview/19441991/preview.jpg)](https://melbourne.figshare.com/articles/UnimelbCorridorSynthetic_zip/10930457)

## Initilal and fine-tuned weight files
The initial weight file (GoogleNet V1 trained on the Places dataset) and the fine-tuned model weights can be found [here](https://melbourne.figshare.com/articles/GoogleNet_weights_trained_on_the_Places_dataset_for_Keras_/10959350). The following are the details of the fine-tuned weight files:

- SynCar - Weights of model fine-tuned on Synthetic Cartoonish images.
- SynPhoReal - Weights of model fine-tuned on Synthetic photo-realistic images.
- SynPhoRealTex - Weights of model fine-tuned on Synthetic photo-realistic textured images.
- GradmagSynCar - Weights of model fine-tuned on synthetic gradmag of SynCar images.
- EdgeRender - Weights of model fine-tuned on Synthetic edge render images.

Other details in the name of the weight files describes the parameters, such as window length, learning rate, batch, ...., etc.

If you are using the dataset or any part of the code, please cite our works: 
- Acharya, D., Khoshelham, K., and Winter, S., 2019. BIM-PoseNet: Indoor camera localisation using a 3D indoor model and deep learning from synthetic images. ISPRS Journal of Photogrammetry and Remote Sensing. 150: 245-258.
- Acharya, D., Singha Roy, S., Khoshelham, K. and Winter, S. 2019. Modelling uncertainty of single image indoor localisation using a 3D model and deep learning. In ISPRS Annals of Photogrammetry, Remote Sensing & Spatial Information Sciences, IV-2/W5, pages 247-254.
