# Introduction

This is a study in which the use of deep learning-based classifiers is observed to distinguish the highlights from COVID-19 radiological images such as CXR and CT scans, and prepare them from a combination of Pneumonia, other respiratory illnesses, and common cases.
The model uses a **Grad-CAM** approach to create class enactment maps. 
With this, a radiologist will be able to use our model both freely and closely. 
Because of the patients with Pneumonia, the model also focuses on False Positives (FP).
If a patient with Pneumonia is incorrectly identified as COVID-19 positive by our model, the patient will be contacted to be advertised in a COVID-19 area with an appropriate emergency clinic.

# General Architecture

The working of our model is as follows:
- First, images from 2 different datasets are taken as the input of our systems.
- Then basic transformations are applied on the image. (Converting them to black and white, and resizing them).
- Train-test split
- Then features are extracted from the image using convolutional layers
- The pixels are then shredded using pooling layers
- All the features then will be analyzed using fully-connected layer
- The interesting part in the image is then emphasized using Grad-CAM which finally gives the output.

![image](https://user-images.githubusercontent.com/56152405/163548666-7f97751a-023c-4c3c-80b7-236eeb954af7.png)

# Review on various schemes

*Image acquisition:* 
Image acquisition was done using CT-scans and Chest X-Ray images acquired from 3 datasets which are: COVID-chest X-ray-dataset, SARS-COV-2 CT-scan, Chest X-ray Images (Pneumonia).
ref. [41],[14],[3]
*Feature Extraction:*
Convolutional layers extracts feature from the image.
ref. [24],[25],[29]
*Image Pre-processing:*
Image Pre-processing is done in the pooling layer which further reduces the feature.
ref. [17],[12],[11]
*ROI from NON-ROI*
Grad-CAM separates ROI from non-ROI
ref. [36],[35],[34]
*Classification*
The fully connected layer does the feature analysis and it then passes output to GRAD-CAM which further focuses on ROI.
ref. [29],[12],[33]

|     **Authors**                                    |     **Methodology   or Techniques used**                                           |     **Advantages**                                                                                                                                              |     **Issues**                                                                                                            |     Metrics   used                                   |
|----------------------------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
|     Group I:                                       |                                                                                    |                                                                                                                                                                 |                                                                                                                           |                                                      |
|     Vijay Badrinarayanan et al. [2017]             |     Deep NN architecture                                                           |     Performs non-linear up-sampling.                                                                                                                            |     Training time is significantly high                                                                                   |     mIoU metric                                      |
|     Giacomo Capizzi et al. [2019]                  |     Fuzzy system combined with a neural   network                                  |     Internal organs examined by the use of   screening methods.                                                                                                 |     FN ratio was high                                                                                                     |     Accuracy                                         |
|     Retz Mahima Devarapalli et al. [2019]          |     Support Vector Machine                                                         |     Detection and confirmation of early   detection of cancer.                                                                                                  |     Except SVM other models have low   accuracy.                                                                          |     Sensitivity                                      |
|     Mohammad Jamshidi et al. [2018]                |     Deep Learning                                                                  |     Accelerate the process of diagnosis of   the COVID-19 disease.                                                                                              |     Novel approaches needed for problems of this level of complexity.                                                     |     Accuracy                                         |
|     Qiao Ke et al. [2019]                          |     Neuro-heuristic approach                                                       |     The method is flexible and has low   computational burden.                                                                                                  |     Improvements in the developed methods                                                                                 |     Accuracy                                         |
|     Michelle Livne et al. [2019]                   |     U-Net                                                                          |     Proposed alternative for classic   rule-based segmentation algos                                                                                            |     Improved segmentation and methodologies   to deal with specific pathologies.                                          |     Dice coefficient                                 |
|     Jingdong Yang et al.     [2021]                |     Dilated MultiResUNet                                                           |     superior accuracy and achieve better   generalization performance                                                                                           |     more test required to improve the system                                                                              |     Dice                                             |
|     Michał Wieczorek et al. [2020]                 |     NN powered predictor                                                           |     Unified architecture that makes it easy   to work as predictor                                                                                              |     Low network efficiency when amount of   data is low                                                                   |     Accuracy                                         |
|     Pranav, Jothi V et al. [2020]                  |     Deep Learning                                                                  |     The model has achieved high sensitivity   specificity                                                                                                       |     Larger dataset required for higher accuracy                                                                           |     Specificity                                      |
|     Kanakaprabha. S et al.     [2021]              |     Deep Learning                                                                  |     Even with abnormal condition prediction   can be done                                                                                                       |     Ensembling of different models to   improve the accuracy                                                              |     Accuracy                                         |
|     Harsh Agrawal et al. [2020]                    |     Deep Learning                                                                  |     Image processing can be used to improve   the performance of deep learning models                                                                           |     classifiers with more than one model can   be considered                                                              |     F1 score                                         |
|     Gautam R.G. et al. [2021]                      |     Deep Neural Networks with PyTorch                                              |     Proposed reverse-transfer learning                                                                                                                          |     -                                                                                                                     |     mIoU - mean Intersection over Union              |
|     Simona Condaragiu et al. [2021]                |     DNNs                                                                           |     Comparative analysis of different models   and data                                                                                                         |     low quality images harms the system very drastically                                                                  |     F1, AUC                                          |
|     Hayat O. Alasasfeh et al. [2020]               |     Deep learning                                                                  |     Model is fast and accurate at the same   time                                                                                                               |     Large dataset can't be inserted into the   NN at once                                                                 |     F1 score                                         |
|     Md. Foysal et al. [2021]                       |     Ensemble DNNs                                                                  |     Overall false prediction is decreased                                                                                                                       |     High GPU power was required                                                                                           |     F1 score                                         |
|     Mohd. H. Naviwala et al. [2021]                |     DNNs                                                                           |     Performance comparison of different DNNs                                                                                                                    |     Vanishing gradient                                                                                                    |     AUC                                              |
|     Asif Iqbal Khan et al. [2020]                  |     CoroNet, a Deep Neural Network Model                                           |     Tailored COVID-19 detection through   custom XceptionNet                                                                                                    |     Their model was overfitting the test set.                                                                             |     Classification Accuracy                          |
|     Tulin Ozturk et al. [2020]                     |     DarkCovidNet Model, a neural network   based on customization of DarkNet-19    |     End-to-end architecture which requires   only raw X-ray image to determine CoVID status,                                                                    |     Model made incorrect predictions in poor   quality X-ray images                                                       |     Accuracy                                         |
|     Ezz EL-DIn Hemdan et al. [2020]                |     COVIDX-Net, Framework of various Deep   Neural Networks                        |     Use of Stochastic Gradient Descent for   faster running time                                                                                                |     Batch Gradient Descent would be   drastically better for a dataset of this size, so as to reach global minimum        |     F1-Score                                         |
|     Wei Li et al. [2020]                           |     RNN Hybrid with ResNet                                                         |     Using ResNet in Hybrid with RNN for   faster prediction.                                                                                                    |     RNN based Hybrid requires large labelled   training set, and absence of this is significantly hurting performance     |     T-Test                                           |
|     Group II:                                      |                                                                                    |                                                                                                                                                                 |                                                                                                                           |                                                      |
|     Abbas A et al. [2021]                          |     DeTraC                                                                         |     DeTraC can deal with any irregularities   in the image dataset                                                                                              |     Limited availability of annotated   medical images makes it hard to classify                                          |     sensitivity                                      |
|     Sharvari Kalgutkar et al. [2019]               |     Transfer Learning                                                              |     The work achieves increased accuracy   values by 5–7 %.                                                                                                     |     Patients historical data can be used for   more detailed result                                                       |     Accuracy, F1 score                               |
|     Varalakshmi P. et al. [2o21]                   |     Transfer learning                                                              |     Used sonograms for detection                                                                                                                                |     Sensitive to outliers                                                                                                 |     F1 score                                         |
|     Shuai Wang et al. [2020]                       |     Modified Inception transfer learning   model                                   |     Identifies Region of Interest and then   performs feature extraction and classification in ROI.                                                             |     High number of False Negatives due to   improper acclimation of ROI in cases with Pneumonia                           |     AUC                                              |
|     Y. Pathak et al. [2020]                        |     Deep Transfer Learning                                                         |     Used 10-fold cross-validation to prevent   overfitting                                                                                                      |     Optimal selection of hyper-parameters is   not considered                                                             |     Precision                                        |
|     Muhammad E.H Chowdhury et al. [2020]           |     Transfer learning with Softmax Function                                        |     Softmax activation function enhances   result for multi class classification between pneumonia and covid                                                    |     Artificial dataset created by just   rotation is not helpful to the model at all, and is creating a bias.             |     F1-Score                                         |
|     Neha Gianchandani et al. [2020]                |     Deep transfer learning                                                         |     Improve the generalization capability of   the classifier for binary and multi-class problems.                                                              |     More data can be levered to improve the   feature extraction capabilities model.                                      |     F1-Score                                         |
|     Varan Singh Rohila et al. [2021]               |     Ensemble learning                                                              |     Proposed a new framework to detect traces   of COVID-19                                                                                                     |     Requires 2 or more GPUs to run                                                                                        |     AUC                                              |
|     Stefanus Tao Hwa Kieu et al. [2021]            |     Ensemble      classification                                                   |     Achieved overall accuracy of 97.9o%                                                                                                                         |     The image resolution should be increased                                                                              |     Accuracy                                         |
|          Ojas A. Ramwala et al. [2020]             |     Deep Residual Network                                                          |     solved the issue of accuracy reduction   with the increase in depth of the network                                                                          |     Models can be trained only on balanced   datasets                                                                     |     Accuracy, Positive/Negative Predicted   Value    |
|     Muhammad A.N, et al. [2021]                    |     Attention Based Residual Network                                               |     Diagnostic Learning on Lung Ultrasound                                                                                                                      |     larger dataset could've helped                                                                                        |     Accuracy                                         |
|     Group III:                                     |                                                                                    |                                                                                                                                                                 |                                                                                                                           |                                                      |
|     Michael J. Horry et al. [2020]                 |     Multimodal Imaging                                                             |     The model is capable of classifying   COVID-19 vs Pneumonia conditions                                                                                      |     Data was not sufficient                                                                                               |     F1 score                                         |
|     Ferhat Bozkurt et al. [2021]                   |     LBP (local binary pattern)                                                     |     Over 99% accuracy is achieved with the   LBP+SVM on CXR images                                                                                              |     Model is biased as recall is very low                                                                                 |     Accuracy, precision, recall                      |
|     Rajarshi Bhadra et al. [2020]                  |     Multi-layered pre-processing                                                   |     Accuracy of 99.1% achieved                                                                                                                                  |     Training time increased 1o fold                                                                                       |     F1 score, accuracy                               |
|     Ahmed M.F. et al. [2020]                       |     XGBoost                                                                        |     Model showed promising results for   further studies                                                                                                        |     Training time was reported quite high                                                                                 |     AUC                                              |
|     Fian Yulio Santoso et al. [2020]               |     Batch normalisation                                                            |     Proposed model avoids overfitting                                                                                                                           |     Testing and tuning still required                                                                                     |     Accuracy                                         |
|     R.M. Rawat et al. [2020]                       |     Comparative study                                                              |     Compared different models and proposed   the best among them                                                                                                |     Models weren't tuned                                                                                                  |     F1, accuracy                                     |
|     Buyut Khoirul Umri et al. [2020]               |     Contrast Limited Adaptive Histogram   Equalization                             |     Study to compare CLAHE with existing   models                                                                                                               |     Small dataset of 40 images                                                                                            |     Accuracy                                         |
|     Harsh Panwar et al. [2o2o]                     |     NCOVnet, a Neural Network based Model                                          |     Model prevented bias against train test   split                                                                                                             |     -                                                                                                                     |     F1 Score                                         |
|     Group IV:                                      |                                                                                    |                                                                                                                                                                 |                                                                                                                           |                                                      |
|     P.K. Gupta et al [2020]                        |     Deep Learning using Grad-CAM                                                   |     Gradient Weighted Class Activation   Mapping results in really accurate binary classification between normal and   infected lung scans.                     |     Slightly biased model towards COVID and   Pneumonia due to GRAD-CAM                                                   |     F1-Score                                         |
|     Luca Brunese et al. [2020]                     |     Pipelining of two Visual Geometry Group   Models                               |     1st pipeline classifies between healthy   and infected lungs, 2nd pipeline then identifies COVID                                                            |     Skewed dataset, and heavy processing   needed for prediction                                                          |     F1 Score                                         |
|     Hap Quan et al. [2021]                         |     Detection through DenseCapsNet, a Capsule   Neural Network method.             |     Can perform sound analysis without much   data and any kind of pre-processing                                                                               |     FP score was high for Pneumonia, which   results in low F1 score.                                                     |     Sensitivity, Specificity                         |
|     Ali Narin [2021]                               |     Specific feature selection which is then   passed on custom SVM based Model    |     Used Ant Colony Optimization technique   using results from SVM and KNN to customize SVM                                                                    |     Overfitting due to mini batch size of 10   and only 30 iterations of Neural Model                                     |     Accuracy                                         |
|     Warda M. Shaban et al. [2020]                  |     Enhanced KNN based Classifier                                                  |     Used Hybrid selection on extracted   features to plot cluster space and hence classifying on the basis of vector   being nearest to the specific cluster    |     Extensive feature extraction is needed,   and large dataset is also needed to plot COVID- non COVID cluster space     |     Accuracy                                         |
|     Ferhat Ucar et al. [2020]                      |     COVIDiagnosis-Net, a Deep   Bayes-SqueezeNet based model                       |     Extremely small model size, makes   hardware deployment easily feasible.                                                                                    |     Generated data tuples by adding shear and   brightness to the original x-ray which is just adding additional bias.    |     Accuracy                                         |

* Group-I
In group-I, researchers used Deep Neural Network as their major component. They have worked majorly on feature selection and classification. Asif Iqbal Khan et al. [32] have used CNN for feature selection which proved to be quite helpful in comparison to Wei Li et al. [34] and Ali Narin [38] as they have used RNN and SVM based models respectively.
* Group-II
In group-II, researchers used transfer learning as their major component. In this group, Sharvari Kalgutkar et al. [42] achieved accuracy values by 5-7% but found dataset scarcity as an issue. Also, Varan Singh Rohila et al. [16] proposed a new framework to detect the virus.
Overall, these models proved to be better at dealing with outliers, distorted and low-resolution images (except for Varalakshmi P. et al. [29], who used Sonograms for detection).
* Group-III
Here in Group-III, most of the papers were related to Image Pre-Processing, and important papers in this section were Michael J. Horry et al. [14], Ferhat Bozkurt et al. [21] and  Buyut Khoirul et al. [30], They had used CLAHE (Contrast Limited Adaptive Histogram Equalization), LBP (Local Binary Pattern) and Multimodal Imaging respectively, to better understand the data, by comparing with adjacent pixels.
* Group-IV
Papers in Group-IV were mostly based around Region of Interest (ROI) detection and feature extraction, few Note-worthy papers here were P.K. Gupta et al. [36] and Luca Burnese et al. [33], They elegantly showed application of ROI identification through Grad-CAM and VGG, and their accuracy regarding infected and healthy lungs were among the top, then one another important paper was Warda M. Shaban et al. [38] extracted features through a combination of SVM and KNN based models.

# Conclusions
*Among various schemes which is best for which application*

**GROUP-I**
Feature Extraction and Reduction + Classifier
Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance to various objects in the image and be able to differentiate one from the other. CNN is the best model for image processing so we have chosen CNN for our objective.
An input layer, a hidden layer, and an output layer make up a convolutional neural network. Any middle layer in a fed-forward neural network is considered to be hidden since the activation function and the final convolution hide their inputs and outputs. The hidden layers in a convolutional neural network are the layers that perform the convolution. Typically, this entails a layer that performs the dot product of the convolution kernel with the input matrix of the layer. The activation function of this product is normally ReLU, and it is generally a Frobenius inner product. The convolution operation creates a feature map as the convolution kernel slides along the input matrix for the layer. This contributes to the input of the next layer. other layers, such as pooling layers, completely connected layers, and normalisation layers, come next. 17 Except for convolution operations that take place in one or more layers of a CNN, CNNs are quite similar to vanilla neural networks. 

**Group-IV**
ROI Identification and Analysis
Gradient-weighted Class Activation Mapping (Grad-CAM) utilizes the gradients of any target feature flowing into the final convolutional layer to produce a coarse localization map emphasising the essential regions in the image for predicting the concept.
Softmax is a mathematical process that converts a vector of integers into a vector of probabilities, with the probabilities of each value proportional to the vector's relative scale. The softmax function is most commonly used in applied machine learning as an activation function in a neural network model. The network is set up to produce N values, one for each class in the classification task, and the softmax function is used to normalise the outputs, converting them from weighted sum values to probabilities that add up to one. 

ISSUES IN EXISTING SYSTEMS
* Previous systems simply classified the input.
  * Existing systems simply classified the users whether they are CoVID positive or not. They don't provide any further knowledge in that direction
* More data can be levered to improve the overfitting issue.
  * As we have seen previously most of the papers had issue of overfitting due to lack of data availability, and they have also mentioned that they had issues while training their models due to un-availability of data.
* Training and testing time were significantly high.
  * It was also observed in some papers due to ongoing pandemic most of the papers were published their work immediately without any proper tuning and testing.

# Module descriptions 
**Grad-CAM:**
Gradient-weighted Class Activation Mapping (Grad-CAM) utilizes the gradients of any target 
feature flowing into the final convolutional layer to produce a coarse localization map emphasizing the essential regions in the image for predicting the concept.
Grad-CAM can be used for weakly-supervised localization, i.e., determining the location of particular objects using a model that was trained only on whole-image labels rather than explicit location annotations.

![image](https://user-images.githubusercontent.com/56152405/163549699-d571c65e-81ed-49b0-adcc-f37aaba60909.png)

**Convolutional Neural Network (CNN):**
An input layer, a hidden layer, and an output layer make up a convolutional neural network. Any middle layer in a fed-forward neural network is considered to be hidden since the activation 
function and the final convolution hide their inputs and outputs. The hidden layers in a convolutional neural network are the layers that perform the convolution. Typically, this entails a layer that performs the dot product of the convolution kernel with the input matrix of the layer. The activation function of this product is normally ReLU, and it is generally a Frobenius inner product. The convolution operation creates a feature map as the convolution kernel slides along the input matrix for the layer. This contributes to the input of the next layer. other layers, such as pooling layers, completely connected layers, and normalization layers, come next.
Except for convolution operations that take place in one or more layers of a CNN, CNNs are quite similar to vanilla neural networks. 

* **VGG-16**
VGG16 is a convolutional neural network model developed by K. Simonyan and A. Zisserman of the University of oxford in their paper "Very deep convolutional networks for large-scale image 
recognition." In ImageNet, a dataset of over 14 million images belonging to 1ooo classes, the model achieves 92.7 percent top-5 test accuracy. It was one of the well-known models shown at the ILSVRC-2o14. It improves on AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layers, respectively) with several 33 kernel-sized filters one 

<p align="center">
  <img src="https://user-images.githubusercontent.com/56152405/163550220-6dfe4b25-a757-47f6-9cb2-65fdeb5320ea.png" />
</p>

Unfortunately, VGG.net has two significant flaws:
- Training is very slow.
- The network architecture weight (in relation to disk/bandwidth) is rather large.
VGG16 is over 533MB in size because to its depth and the number of completely connected nodes. This makes VGG deployment a difficult process.

We used VGG16 with imagenet weights and all the internal layers with their pre-trained weights in our project. To get the outcome, we used pooling, densification, dropout, and a flattened layer. A softmax function is also used in the final layer. The model was created with Adam as the optimizer and accuracy as the optimization matrix.

* **DENSE NET121**
DenseNet (Dense Convolutional Network) is an architecture that employs fewer connections 
between layers to make deep learning networks increasingly deeper while also making them more economical to train. DenseNet is a convolutional neural network with each layer connected to all other layers deep in the network, for example, the first layer is connected to the second, third, fourth, and so on; the second layer is connected to the third, fourth, fifth, and so on. This is done to guarantee that the greatest amount of data can move between the levels of the network. It does not integrate features by putting them together, like ResNets do, but rather by combining them. The 'ith' layer, for example, has an I input and includes all of the previous convolutional blocks' feature maps. 

All succeeding 'eye-eye' layers get their own feature maps. It uses the '(I(I+1))/2' connection to link the network, rather than the 'I' connection used in normal deep learning designs. Because no unnecessary feature mappings must be trained, it uses fewer parameters than typical convolutional neural networks.

<p align="center">
  <img src="https://user-images.githubusercontent.com/56152405/163550451-c12a6e7e-eab0-4321-a258-9942ce70168f.png" />
</p>

Traditional fed-forward neural networks link the outputs of the previous layer to the next layer after conducting a composite of operations. 

We can look at feature maps as network data. Each layer has access to its previous feature maps and, as a result, to the collective knowledge. Then, in a feаture map of solid k information, each layer adds a new piece of information to the collection

<p align="center">
  <img src="https://user-images.githubusercontent.com/56152405/163550483-3f14d387-c777-4593-a6a2-6feb4ffd14da.png" />
</p>

DenseNets uses a fundamental connection rule to merge the qualities of identity mapping, deep observation, and various depths. They allow network features to be reused, resulting in more compact and, in our experience, more accurate model learning. Because of their compact internal representations and minimal feature redundancy, DenseNets can be useful feature extractors for a variety of computer vision applications based on convolutional features

* **RESNET50**
ResNet, or Residual Networks, is a traditional neural network that is used as a backbone in many computer vision applications. The winner of the ImageNet challenge in 2015 was this model. The fundamental breakthrough with ResNet was that it allowed us to successfully train extremely deep neural networks with 150+ layers. Due to the problem of vanishing gradients, training very deep neural networks was challenging prior to ResNet.

However, just stacking layers together will not increase network depth. Because of the infamous disappearing gradient problem, which occurs when a gradient is back-propagated to earlier layers, repeated multiplication can cause the gradient to become extremely tiny. As a result, as the network becomes more complex, its performance becomes saturated or even degrades fast.

The concept of skip connection was initially introduced by ResNet. The graphic below shows how to skip a connection. The figure on the left shows stacking convolution layers one on top of the other. on the right, we stack convolution layers as before, but now we additionally add the original input to the convolution block's output. This is referred to as a skip connection

The reason why skipping connections works is because:
- They allow the model to learn an identity function, ensuring that the higher layer will perform at least as well as the lower layer, if not better.
- They mitigate the problem of disappearing gradient by enabling gradient to flow along an alternate shortcut channel.

The ResNet-50 model is made up of five stages, each having its own convolution and identity block. Each convolution block contains three convolution layers, and each identity block contains three as well. over 23 million trainable parameters make up the ResNet-50.

# DATASET DISCRIPTION
**COVID-chest X-ray-dataset** – This dataset was compiled by Cohen et al. and is widely utilised by numerous scholars. The database is kept up to date with the most recent updates. This dataset has 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images.
database of COVID-19 x-rays images from the Italian Society of Medical and Interventional Radiology (SIRM) COVID-19 DATABASE, Novel Corona Virus 2019 Dataset

**Chest X-ray Images (Pneumonia)** – This dataset was developed by Kermany et al. A substantial number of OCT and CXR images are publically available in this dataset. We used the X-Rays section of this dataset in the studies, which included 5863 photos of Pneumonia and Normal individuals. 

# RESULTS AND DISCUSSION
We trained all the models for 15 epochs and used accuracy and loss as the optimization matrix, and then produced the confusion matrix to get more intuitive results. 

VGG16:
<p align="center">
  <img src="https://user-images.githubusercontent.com/56152405/163551310-b026dc05-8e07-439d-b92c-005ed57a584b.png" />
</p>

DenseNet121:
<p align="center">
  <img src="https://user-images.githubusercontent.com/56152405/163551327-3296f755-554a-47c2-9b20-35aae1a8a87f.png" />
</p>

ResNet50:
<p align="center">
  <img src="https://user-images.githubusercontent.com/56152405/163551413-655e5fa0-51b7-4734-83f7-e01df1685a5b.png" />
</p>

Model Metrics:

|     Model          |     Precision    |     Recall    |     Accuracy    |     F-Score    |
|--------------------|------------------|---------------|-----------------|----------------|
|     VGG16          |     0.9643       |     1.000     |     0.9825      |     0.9818     |
|     DenseNet121    |     0.7143       |     1.000     |     0.8596      |     0.8333     |
|     ResNet50       |     0.6786       |     1.000     |     0.8421      |     0.8085     |

For producing visual explanations for our Convolutional Neural Network (CNN) –based models we use Grad – CAM based heatmaps. Below are the heatmaps produced by each model for the same image of the class: diagnosed as CoVID-19. Note that these heatmaps are produced from weights calculated by the last convolutional layer in each network:

* VGG-16 (Non-CoVID)

**Inference:** we can see that the last layer of VGG-16 is not focusing on lung tissues because it’s a Non-CoVID case. 

ex 1
![image](https://user-images.githubusercontent.com/56152405/163551825-1297d323-3af1-49f1-b457-78c22e4ef715.png)

ex 2
![image](https://user-images.githubusercontent.com/56152405/163551836-b9f73aa8-a1dc-41ac-922c-6cefb055c765.png)

ex 3
![image](https://user-images.githubusercontent.com/56152405/163551876-5d49ed34-2f02-48f4-bc99-9c9fbaadc317.png)

Inference: 
In the picture below, GRAD CAM can focus on the actual ROI, hence proving the of reliability and efficiency of VGG-16
![image](https://user-images.githubusercontent.com/56152405/163551892-06ba212d-f570-4ebd-9a6e-00ecb4024978.png)

Inference:
For ResNet50 and Dense121 we can observe that both the models’ last layer cannot identify the actual ROI hence they are less reliable than VGG-16
ResNet50 (CoVID-19)
![image](https://user-images.githubusercontent.com/56152405/163551957-dff09083-e84d-4852-ac9c-bc4964a74e03.png)

DenseNet121 (CoVID-19)
![image](https://user-images.githubusercontent.com/56152405/163551977-cc90f1a6-7de8-4bd9-82c5-5f108f9e20db.png)

Comparative Study
Basic comparative studies against the paper which we found similar and relevant to out project
|     Parameters               |     [19]                               |     [31]                     |     Proposed                                                                  |
|------------------------------|----------------------------------------|------------------------------|-------------------------------------------------------------------------------|
|     Dataset                  |     COVID-19   Radiography Database    |     -No longer available-    |     COVID-19   Radiography Database,      Chest   X-Ray Images (Pneumonia)    |
|     Training – Test Ratio    |     80:20                              |     70:30                    |     80:20                                                                     |
|     Validation Accuracy      |     -                                  |     -                        |     96.77%, 85.96%, 84.21%                                                    |
|     Testing Accuracy         |     97.69%                             |     98.82                    |     98.25%, 98.16%, 96.31%                                                    |
|     Precision                |     -                                  |     0.95                     |     0.9643, 0.7143, 0.6786                                                    |
|     F-Score                  |     -                                  |     0.94                     |     0.9818, 0.8333, 0.8085                                                    |
|     Recall                   |     -                                  |     0.98                     |     1.0, 1.0, 1.0                                                             |
|     Epochs                   |     45                                 |     20                       |     15                                                                        |
|     Model                    |     ResNet32                           |     nCOVnet                  |     VGG-16, ResNet50, DenseNet121                                             |

These are the results of training each model for 15 epochs:
VGG-16:

|     Epoch    |     Train loss    |     Train accuracy    |     Validation loss    |     Validation accuracy    |
|--------------|-------------------|-----------------------|------------------------|----------------------------|
|     1        |     0.6933        |     0.5945            |     0.5562             |     0.9825                 |
|     2        |     0.5609        |     0.7742            |     0.4504             |     0.9825                 |
|     3        |     0.4902        |     0.8618            |     0.376              |     0.9649                 |
|     4        |     0.4151        |     0.8894            |     0.3016             |     0.9825                 |
|     5        |     0.3533        |     0.9032            |     0.2553             |     0.9825                 |
|     6        |     0.2929        |     0.9447            |     0.198              |     0.9825                 |
|     7        |     0.2413        |     0.9493            |     0.1811             |     0.9825                 |
|     8        |     0.2111        |     0.9677            |     0.1377             |     0.9825                 |
|     9        |     0.199         |     0.9539            |     0.1165             |     0.9825                 |
|     10       |     0.151         |     0.977             |     0.0957             |     1                      |
|     11       |     0.1425        |     0.9724            |     0.0856             |     1                      |
|     12       |     0.1425        |     0.9677            |     0.1043             |     0.9825                 |
|     13       |     0.1403        |     0.9585            |     0.0712             |     0.9825                 |
|     14       |     0.1085        |     0.977             |     0.0665             |     1                      |
|     15       |     0.1204        |     0.9677            |     0.0636             |     0.9825                 |

DenseNet121:

|     Epoch    |     Train loss    |     Train accuracy    |     Validation loss    |     Validation accuracy    |
|--------------|-------------------|-----------------------|------------------------|----------------------------|
|     1        |     0.617         |     0.765             |     0.6872             |     0.8246                 |
|     2        |     0.4655        |     0.8479            |     0.6769             |     0.8246                 |
|     3        |     0.3596        |     0.9124            |     0.6624             |     1                      |
|     4        |     0.3033        |     0.9263            |     0.6743             |     0.5088                 |
|     5        |     0.2134        |     0.9493            |     0.6674             |     0.5088                 |
|     6        |     0.1831        |     0.9539            |     0.5873             |     0.8596                 |
|     7        |     0.1631        |     0.9631            |     0.5984             |     0.5263                 |
|     8        |     0.1365        |     0.9724            |     0.6381             |     0.5088                 |
|     9        |     0.1436        |     0.9585            |     0.5955             |     0.5088                 |
|     10       |     0.1005        |     0.9954            |     0.5056             |     0.7368                 |
|     11       |     0.1222        |     0.9724            |     0.5523             |     0.5789                 |
|     12       |     0.1287        |     0.9493            |     0.6406             |     0.5088                 |
|     13       |     0.121         |     0.9724            |     0.4376             |     0.7719                 |
|     14       |     0.0973        |     0.9677            |     0.4272             |     0.7719                 |
|     15       |     0.0734        |     0.9816            |     0.3509             |     0.8596                 |

ResNet50:

|     Epoch    |     Train loss    |     Train accuracy    |     Validation loss    |     Validation accuracy    |
|--------------|-------------------|-----------------------|------------------------|----------------------------|
|     1        |     0.61          |     0.6728            |     0.6929             |     0.5088                 |
|     2        |     0.4121        |     0.8618            |     0.6808             |     0.5088                 |
|     3        |     0.3217        |     0.894             |     0.6786             |     0.5088                 |
|     4        |     0.2702        |     0.9309            |     0.6692             |     0.5088                 |
|     5        |     0.2338        |     0.9217            |     0.6332             |     0.5088                 |
|     6        |     0.1939        |     0.9401            |     0.6384             |     0.5088                 |
|     7        |     0.1684        |     0.9309            |     0.6245             |     0.5088                 |
|     8        |     0.1894        |     0.9355            |     0.6297             |     0.5088                 |
|     9        |     0.1499        |     0.9493            |     0.6738             |     0.5088                 |
|     10       |     0.1469        |     0.9539            |     0.5862             |     0.5263                 |
|     11       |     0.1311        |     0.9677            |     0.5197             |     0.6491                 |
|     12       |     0.1434        |     0.9585            |     0.5895             |     0.5439                 |
|     13       |     0.1506        |     0.9585            |     0.4586             |     0.7193                 |
|     14       |     0.1139        |     0.9585            |     0.4466             |     0.7193                 |
|     15       |     0.1209        |     0.9631            |     0.3773             |     0.8421                 |

Conclusion and Future Work

In this work, we used binary image classification to identify CoVID-19 and non-CoVID19 positive patients.
According to the study, non-CoVID-19 positive patients may have Pneumonia or other respiratory problems.
Several experiments have employed CXR and CT-Scan images of the chest to detect CoVID-19 patients.

In order to make our deep learning model more interpretable and explainable, we used a color visualisation method utilizing the GRAD-CAM technique in our trials which could be further used in other research or acedemia purposes, for eg, using them in a bigger application or passing them in a pipeline.

# References
[1]Neural network powered COVID-19 spread forecasting model

[2]Performance analysis of lightweight CNN models to segment infectious lung tissues of COVID-19 cases from tomographic images

[3]COVID-19 Detection Through Transfer Learning Using Multimodal Imaging Data

[4]COVID 19, Pneumonia and Other Disease Classification using Chest X-Ray images

[5]COVID-19 Diagnosis from Chest Radiography Images using Deep Residual Network

[6]SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

[7]Small Lung Nodules Detection Based on Fuzzy-Logic and Probabilistic Neural Network With Bioinspired Reinforcement Learning

[8]Lung Cancer Detection of CT Lung Images

[9]Artificial Intelligence and COVID-19: Deep Learning Approaches for Diagnosis and Treatment

[1o]A neuro-heuristic approach for recognition of lung diseases from X-ray images

[11]A U-Net Deep Learning Framework for High Performance Vessel Segmentation in Patients With Cerebrovascular Disease

[12]Neural network powered COVID-19 spread forecasting model

[13]A Novel Coronavirus from Patients with Pneumonia in China, 2o19

[14]High-Resolution Representations for Labeling Pixels and Regions

[15]Deep High-Resolution Representation Learning for Human Pose Estimation

[16]Deep Learning Assisted Covid-19 Detection using full CT-scans

[17]A Survey of Deep Learning for Lung Disease Detection on Medical Images: State-of-the-Art, Taxonomy, Issues and Future Directions

[18] COVID-19 Pandemic in India: Through the Lens of Modeling

[19] Attention Based Residual Network for Effective Detection of COVID-19 and Viral Pneumonia

[2o] Evaluation of Convolutional Neural Networks for COVID-19 Detection from Chest X-Ray Images

[21] Local Binary Pattern Based COVID-19 Detection Method Using Chest X-Ray Images

[22] Covid Detection from CXR Scans using Deep Multi-layered CNN

[23] Using CNN-XGBoost Deep Networks for COVID-19 Detection in Chest X-ray Images

[24] Application of Deep Learning for Early Detection of COVID-19 Using CT-Scan Images

[25] COVID-19 Detection using Convolutional Neural Network Architectures based upon Chest X-rays Images

[26] Deep Learning Approach for COVID-19 Detection Based on X-Ray Images

[27] Automatic Detection of COVID-19 from Chest X-ray Images with Convolutional Neural Networks

[28] Performance Analysis of Deep Learning Frameworks for COVID 19 Detection

[29] A Transfer Learning Model for COVID-19 Detection with Computed Tomography and Sonogram Images

[3o] Detection of Covid-19 in Chest X-ray Image using CLAHE and Convolutional Neural Network

[31]Application of deep learning for fast detection of COVID-19 in X-Rays using nCOVnet

[32]CoroNet: A deep neural network for detection and diagnosis of COVID-19 from chest x-ray images

[33]Novel Coronavirus Identification from ChestX-Ray Images By Using VGG

[34]NIA-Network: Towards improving lung CT infection detection for COVID-19 diagnosis

[35]A new COVID-19 Patients Detection Strategy (CPDS) based on hybrid feature selection and enhanced KNN classifier

[36]A deep learning and grad-CAM based color visualization approach for fast detection of COVID-19 cases using chest X-ray and CT-Scan images

[37]DenseCapsNet: Detection of COVID-19 from X-ray images using a capsule neural network

[38]Accurate detection of COVID-19 using deep features based on X-Ray images and feature selection methods

[39]Automated detection of COVID-19 cases using deep neural networks with X-ray images

[4o]A deep learning algorithm using CT images to screen for Corona Virus Disease (COVID-19)

[41]Deep Transfer Learning Based Classification Model for COVID-19 Disease

[42]Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection Using Chest X-ray

[43]COVIDX-Net: A Framework of Deep Learning Classifiers to Diagnose COVID-19 in X-Ray Images

[44]COVIDiagnosis-Net: Deep Bayes-SqueezeNet based diagnosis of the coronavirus disease 2o19 (COVID-19) from X-ray images

[45]Classification of the COVID-19 infected patients using DenseNet2o1 based deep transfer learning
