# NTIRE2021-IQA-MACS



# Training Networks From Scratch

## Preparing Datasets

Download datasets: <br/>
[1] [TID 2008 dataset](http://www.ponomarenko.info/tid2008/tid/tid2008.rar) (~550 MB) <br/>
[2] [TID 2013 dataset](http://www.ponomarenko.info/tid2013/tid2013.rar) (~908 MB) <br/>
[3] [PieAPP](https://web.ece.ucsb.edu/~ekta/projects/PieAPPv0.1/all_data_PieAPP_dataset_CVPR_2018.zip) (~2.2 GB) <br/>
[4] PIPAL dataset <br/>
<br/>
Please Make all the formats in the reference images of TID datasets in Upper Case (.BMP) (Typically I25 in the reference images has .bmp format instead of .BMP) <br/>
Set datasets directories in [`create_data_pieapp_tid.py`](./prepare_dataset/create_data_pieapp_tid.py) and run it: <br/> 

      python3 prepare_dataset/create_data_pieapp_tid.py

Set PIPAL dataset directory in [`create_data_pipal.py`](./prepare_dataset/create_data_pipal.py) and run it: (Needs at least 16 GB RAM) <br/> 

      python3 prepare_dataset/create_data_pipal.py
      
We can not directly include spearman loss function to train the network, spearman's rank correlation coefficient formula is:
<br/>
<img src="https://latex.codecogs.com/gif.latex?Spearman(y,\hat{y})=1-\frac{6(\|rank(y)-rank(\hat{y})\|^2))}{d(d^2-1)}" /> 
<br/>
It is a non-differentiable function because of the ranking operation. Instead, we train a network to learn sorting the inputs and include this surrogate metric in the loss function:
<br/>
<img src="https://i.imgur.com/Lxvv6Dh.png" /> 
<br/>

## Training Surrogate Ranking Model (Surrogate Spearman Loss Model)
Set total number of training data in  [`train.py`](./ranking_model/train.py) and run it: (Default 2M data) <br/> 

      python3 ranking_model/train.py

(Optional) Now, you can run [`test.py`](./ranking_model/test.py) to see the diffrence in the surrogate spearman and the true spearman for some generated data:

      python3 ranking_model/test.py

Note that, the model input and output for the surrogate ranking model is 16, which is same as the training batch size of the IQA model.

## Training Models (including pretraining models on PieAPP and TID datasets)
1) Using the **configs** text files to train the models using [`train_simple.py`](./train_simple.py) for the basic model architecture, [`train_bn.py`](./train_bn.py) for the model architecture with batch normalization, [`train_attention.py`](./train_attention.py) for the model architecture with attention mechanism and residual blocks and [`train_tiled.py`](./train_tiled.py) for model architecture which splits the image into square tiles and pass them to a ConvLSTM2D layer. The model architectures are shown in the following figures:
Basic Model architecture:
![Alt text](./figures/architecture_simple.png?raw=true "Basic architecture")
Model architecture with attention layer and residual blocks:
![Alt text](./figures/architecture_attention.png?raw=true "attention architecture")
Model architecture with tiling the input and pass it to a ConvLSTM2D:
![Alt text](./figures/architecture_convLSTM.png?raw=true "LSTM architecture")

## References
[1] N. Ponomarenko, V. Lukin, A. Zelensky, K. Egiazarian, M. Carli, F. Battisti, "TID2008 - A Database for Evaluation of Full-Reference Visual Quality Assessment Metrics", Advances of Modern Radioelectronics, Vol. 10, pp. 30-45, 2009. <br/>
[2] N. Ponomarenko, L. Jin, O. Ieremeiev, V. Lukin, K. Egiazarian, J. Astola, B. Vozel, K. Chehdi, M. Carli, F. Battisti, C.-C. Jay Kuo, Image database TID2013: Peculiarities, results and perspectives, Signal Processing: Image Communication, vol. 30, Jan. 2015, pp. 57-77. <br/>
[3] PieAPP: Perceptual Image-Error Assessment Through Pairwise Preference, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. <br/>
[4] PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual Image Restoration , Jinjin Gu, Haoming Cai, Haoyu Chen, Xiaoxing Ye, Jimmy Ren, Chao Dong
ECCV 2020  
