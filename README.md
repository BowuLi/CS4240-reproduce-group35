# Reproduction of: "Restoring Extremely Dark Images in Real Time"

*Original Authors: Mohit Lamba, Kaushik Mitra*

*Reproduction Authors: 
Ting Liang, 5140331, t.liang@student.tudelft.nl
Zhaofeng Shen, 5404568, Z.shen-8@student.tudelf.nl
Bowu Li, 5463351, B.Li-15@student.tudelft.nl
Shixun Wu, 5139414，S.Wu-11@student.tudelft.nl*

## Section I: Introduction

We will present our implementation of reproduce the deep learning research "Restoring Extremely Dark Images in Real Time" [^REDI] in this blog. We're undertaking this as part of a student project for Delft University of Technology's Deep Learning course.

The solution for low-light enhancement It's a method of enhancing an image's overall brightness without sacrificing image size or actual color details. Not only should a practical solution be computationally and memory efficient, but it should also provide intuitive visual restoration. This research proposes a new deep learning architecture for single image restoration in extreme low-light conditions.

## Section II: Theory

In the past few years, restoring dark images is a hot issue in the field of computer vision, and many excellent methods have been proposed. Notably, SID successfully recovered extremely low-light images taken under near-zero lux conditions. Despite its excellent performance, such solutions may not be suitable for practical deployment due to their high computing costs. In this paper, the authors propose a new, lightweight, deep learning structure that allows the restoration perception at par with state-of-the-art methods, and the speed is enhanced by 5-100 times.

![](https://i.imgur.com/8RTld91.png)

Most restore networks use u-Net style encoders and decoders, where processing at lower scales causes significant latency and computational overhead. To alleviate this problem, one can either accelerate hardware operations or reduce network operations and sacrifice restoration quality. The proposed architecture design with architectural parallelism allows concurrent processing of various scale spaces, achieving an additional 30% speedup but with no effect on the restoration quality.

![](https://i.imgur.com/kmm8a5h.png)

The specific parallel structure proposed in this paper is shown in the figure above. There are three scales in the model -- Lower Scale Encoder (LSE), Medium Scale Encoder (MSE), and Higher Scale Encoder (HSE). Each scale directly uses the down-sampled input image as input for convolution operations, independent of the higher scales. 

Since HSE has the lowest resolution, most of the convolution layers are assigned to HSE. The basic building block of HSE is the modified residual density Block (RDB). The modified RDB* can simultaneously process both rectified and non-rectified output of each convolutional layer and avoid the problem of complete loss of information due to non-linear rectification. After each scale has been calculated, Fuse Block 1 (FB 1) and Fuse Block 2 (FB 2) fuses details from all the scales to generate the restored image.

If the execution time of parallel tasks is nearly the same, parallelism can be best utilized to minimize idle time. Therefore, LSE, MSE and HSE need to be executed at the same speed as possible. How do we quantify the speed? A simplified assumption is made to translate this problem to LSE, MSE, and HSE should have the same number of floating-point operations. Through dedicated designed convolutional layer parameters, the model achieves effective parallelism.

In the experiment, Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) were used as evaluation criteria. PSNR is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation[^PSNR]. And SSIM is used for measuring the similarity between two images[^SSIM]. The higher these two indicators are, the closer the restored image is to the original.




## Section III: Reproduction

In order to reproduce the results of the original paper, we trained the model using the full dataset (images captured by Sony 7S II) for one million iterations, as suggested by the paper. The images used for training were all of 0.1s exposure time, with the corresponding images of 10s exposure time as ground truth. When testing the performance of the model, a different set of images with an exposure time of 0.1 was used as a test set.



The image below shows the L1_loss decreasing at the beginning and oscillating around 0.025 after about 300,000 training sessions over the course of one million sessions.

![](https://i.imgur.com/R0LSuWr.jpg#pic_center)


Both the PSNR and SSIM values are used to measure the similarity between the restored image and the Ground truth, the higher the value the more similar it is. The model with one million training iterations has a PSNR of 27.96 and an SSIM of 0.78. We successfully reproduced the model proposed in the paper, compared to the model provided by the authors, which has a PSNR of 28.66 and an SSIM of 0.790.

![](https://i.imgur.com/6Xm3DT3.png#pic_center)
![](https://i.imgur.com/LfvvPqa.png#pic_center)

Next, we show some images of the restored image compared to the ground truth. We can see that our trained model is very good at restoring extreme-dark images and the difference is barely noticeable to the human eye.

![](https://i.imgur.com/maQ41h1.jpg)
> Restored image 


![](https://i.imgur.com/krdqEVN.jpg)
> Ground Truth



![](https://i.imgur.com/AslTptP.jpg)
> Restored image 


![](https://i.imgur.com/rlRDSaf.jpg)
> Ground Truth



![](https://i.imgur.com/HnAU9X2.jpg)
> Restored image 


![](https://i.imgur.com/vKrFWpO.jpg)
> Ground Truth


## Section IV: Faster training - data preprocessing
The original training script tries to train by extracting data with "Dataloader" lazily. When extracting, data will first be processed by the CPU. While during this process, the CPU will become the bottleneck of the whole model, which is the same as what we found during the training. The CPU utilization is always 100%, and the GPU utilization is quite low, about 30% for the V100 GPU. This is also why reducing the training set does not improve the training speed. 

So, after we completely reproduced the original model with the original dataset, we tried to build a preprocessed dataset. This preprocessed dataset aims to iterate through the Dataloader and build the new dataset with these processed data. Then we save these data to the storage device. We read the file to load the data into memory when we start training.

The following code shows how to save the pre-processed data to a file.

```
dataloader_train = DataLoader(load_data(......))
load = []
for _, img in enumerate(dataloader_train):
    load.append((img[0], img[1]))
torch.save(load, 'sony100.pt')
```

This method significantly improves the training speed to about 70% GPU utilization. As a result, the training time is decreased by about 6 hours.

However, this method also introduces a new problem. We build a file by iterating through the original dataset once, and we get the file with about 700MB. If we want to simulate the training process with 1000000 iterations, it will cost about $1000000/180*700/1024=3797GB$, which is unacceptable. Since we have limited GCP credits during this project, we cannot deploy a VM with quite a high disk capacity and the running time is also limited. So, we will only use the preprocessed dataset of 20% or 100% of the original dataset with only one iteration so that we can reduce the setup time of the data file and try to run as many tests as possible. This will surely generate bad results, but the results of the following modifications are qualitatively analyzed, so this is a worthy trade-off.

A possible improvement of this method is to exploit the parallelization. We can run multiple scripts simultaneously. One of them is the training scripts. At the same time, the rests are used for processing the data. After one data file is used, it will be removed from the storage device. Compared to the approach with setting "num_worker" in Dataloader, this method will keep a high utilization of CPUs. While the Dataloader with multiple workers still cannot load and process data in advance, which will bring volatility in CPU and GPU utilization.


## Section V: Different dataset
To explore the effects of different sizes of training datasets and different training datasets on the proposed model, we conduct a series of experiments as follows.

We first used the data preprocessing method mentioned in the previous chapter and transferred the training dataset. We prepared the following training sets:
- Complete the SID[^SID] Sony dataset(the same as the dataset used in the paper)
- Randomly chosen 20% of the SID Sony dataset
- Complete the Fuji dataset
- Randomly chosen 20% of the Fuji dataset

After the training process had finished, we saved the weight values saved in the last training iteration as the parameters for the final trained network. Then, we modified the original training code accordingly, made sure correct training sets were used and clarified that the same training settings were used, such as the number of iterations.

To verify the performances these models have on the different datasets and the analysis for the model generalization capabilities, we applied the following validation sets:
- Canon 70D
- Canon 700D
- Nikon D850

These three datasets from the ELD[^ELD] dataset are the same as those used by the authors in Table 3 in the paper. 

In the original training, the program will execute tests at the pre-determined iterations and save the intermediate testing results. Here we choose the testing results of the test sets from the same camera as the training set as the base for comparison and analysis. We rewrote the test program to load the trained network from the assigned weight file and specified the test sets. To be in accordance with the original paper, we used the same test function created by the author that was used in the original train program. 


![](https://i.imgur.com/cgzcdHh.png)

From the results of the table, we can conclude that under the current model architecture, the Sony training set is more suitable and will generate a network of better restoration quality. In both 100% and 20% cases, when using the validation set generated by the same camera as the training set, the network trained by Sony has better PSNR and SSIM values. We can also tell from the restored images. The model trained by Sony data has better restoration quality, and images are closer to the grand truth images with enough exposure. For images restored by the Fuji-trained model, some details could be recovered in the shade, but for brighter areas, the restoration is inferior, with large blocks of errors. Besides, When restoring the images, the Fuji-trained model is likely to lose the contrast, definition, and color. This may be due to the fact that each company has a different organization of their own raw files, and with different configurations of the photosensor arrays and color sensors, the metadata of the images from Fuji are stored in a different way and cannot be appropriately processed by this network architecture.

![](https://i.imgur.com/fEdlJqE.jpg)
>Restored images from complete Sony training set(left) and ground truth(right)

![](https://i.imgur.com/aGjltYv.jpg)
>Restored images from complete Fuji training set(left) and ground truth(right)


In terms of the capability of generalization, in fact, the model trained by the Fuji dataset has better performances on the ELD dataset if only by comparing the metric values. Analyzing the restored images can also confirm this observation. The sony-trained model always produces an all-white image for restoration for Canon pictures and always produces a much darker and unrecognizable image for Nikon images. The model trained by Fuji can generate images, although being very low-quality restorations and many noises were added, that have some profiles at high brightness contrast areas recognized and restored from the objects in the original image. But in conclusion, none of these datasets can be effectively restored. We did not managed to successfully reproduce the results of this part in the paper. The table 3 of the paper suggested that the Sony-trained model was able to restore images from ELD dataset. We tried but failed to find if there is special method of testing for ELD dataset, which is also not mentioned in the paper. We found that the ELD dataset were provided in 2 ways, raw files or mdb files. We are not sure if this would make a difference in the later testing.

![](https://i.imgur.com/N23eV3I.jpg)
>Restored images from Canon 70D by Fuji-trained model set(left) and ground truth(right)

For both training datasets, the impact from the sizes of the dataset is consistent, that smaller datasets will result in poorer performance. Larger training sets can provide more sample and allows the network parameters to be trained more sufficiently and therefore result in a better restoration capability. For 3 test sets from the ELD dataset, due to the fact that the networks cannot perform effective restoration, we cannot really carry out effective analysis and comparison for analyzing the impacts of the size of the datasets.

![](https://i.imgur.com/yVgojeS.jpg)
>Restored images from 20% Sony training set(left) and ground truth(right)

![](https://i.imgur.com/Ks5u12T.jpg)

>Restored images from complete Sony training set(left) and ground truth(right)

![](https://i.imgur.com/pppuBIu.jpg)
>Restored images from 20% Fuji training set(left) and ground truth(right)

![](https://i.imgur.com/5N5C3rd.jpg)
>Restored images from complete Fuji training set(left) and ground truth(right)



## Section VI: New Normalization Technology - Group Normalization

Yuxin Wu and Kaiming He proposed a new normalization method -- Group Normalization (GN) [^GN] which performs better than Batch Normalization (BN) when the batch size is small. We wonder to know whether will get better (or similar) results if we apply the GN.

![](https://i.imgur.com/90iEITB.png)

Because the channels of visual representations are not entirely independent, GN divides the channels into groups and computes within each group the mean and variance for normalization. We experimented with the modified model with GN and get the results in the table in Section VI. The experimental results show that GN has no positive effect on PSNR and SSIM.

## Section VII: Model reduction
According to the equation given in the paper[^REDI]:

![](https://i.imgur.com/hRwXx8k.png)

We can calculate the number of operations in the Higher Scale Encoder (HSE) of the model. We find that there are two $n_{HSE}$ terms in this formula, so we think if we could remove one RDB layer, there could be much fewer parameters. 

Based on this idea, we modified the network, and get the network with 780672 parameters. The training speed is a bit faster, about 15 hours. 

|   |100% Sony dataset with standard model|100% Sony dataset with 2 RDB\*s model|100% Sony dataset with GN applied model|20% Sony dataset with GN applied model|
|---|---|---|---|---|
|PSNR|21.06|21.27|18.12|19.40|
|SSIM|0.6226|0.6363|0.4513|0.4827|
|Loss|0.015459|0.015952|0.015881|0.007963|

The table above shows the final metric of different models. As we mentioned above, we use quite small preprocessed datasets to speed up training, so we evaluated the standard model again to set the baseline of the qualitative analysis. With the same dataset, we can notice that the 2 RDB model is slightly better than the standard one. This could be considered a kind of error. But there is also no performance degradation. The possible reason is that the original model is already overfitted that small dataset, while a simplified model reduces that phenomenon. This can also be derived from the fact that the standard model has lower loss but also lower PSNR and SSIM.

## Task division
All group members participated in the reproduction of the original network.
Shixun Wu and Ting Liang were responsible for the first research goal: different dataset.
Zhaofeng Shen and Bowu Li were responsible for model reduction and research and apply GN.

## References

[^REDI]: M. Lamba and K. Mitra, "Restoring Extremely Dark Images in Real Time," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 3486-3496, doi: 10.1109/CVPR46437.2021.00349.

[^ELD]: Wei, Kaixuan, et al. “Physics-Based Noise Modeling for Extreme Low-Light Photography.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, pp. 1–1. Crossref, https://doi.org/10.1109/tpami.2021.3103114.

[^SID]: Chen, Chen, et al. “Learning to See in the Dark.” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, June 2018. Crossref, https://doi.org/10.1109/cvpr.2018.00347.

[^GN]: Yuxin and Kaiming. "Group Normalization." 2019 International Journal of Computer Vision volume, July 2019. 
https://doi.org/10.48550/arXiv.1803.08494

[^PSNR]: Wikipedia. *Peak signal-to-noise ratio*. Retrieved 14 April 2022 from: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

[^SSIM]: Wikipedia. *Structural similarity*. Retrieved 14 April 2022 from: https://en.wikipedia.org/wiki/Structural_similarity



