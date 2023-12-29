# BMT-UNet

Methods: A Boundary-Enhanced Multi-scale U-Net network with a Convolutional Transformer (BMT-UNet) is developed to segment the pericardium. Specifically, the pericardial boundary segmentation is first solved through combing the Multi-Scale (MS) module with Boundary-Enhanced (BE) module; then, the Convolutional Transformer (ConvT) module is used for global context integration and feature fusion to solve the problem of internal holes in the segmented pericardial image and improve accuracy of pericardium segmentation. The volume of EAT is automatically quantified using standard fat threshold with a range of -190 to -30 HU. 

The code for the comparative experiments is in the 'networks' folder, and the latest version of the model code is being organized and will be updated soon
