# Deep Learning-Based Segmentation of Atherosclerotic Plaques and Stenosis in Coronary Angiography

[Main Dataset Link](https://www.kaggle.com/datasets/ahmedaboenaba/coronary-artery-angiograms-zip)
[Best Model](https://drive.google.com/file/d/1ER_K0142FKconjGEXs31KXTMTrNpcCe9/view?usp=sharing)

# Training Result
![App Screenshot](./training_history.png)

## Screenshots of result
![App Screenshot](./assets/1.png)
![App Screenshot](./assets/2.png)
![App Screenshot](./assets/3.png)


### How to use
- Clone repo
- Download [Stenosis with masks](https://drive.google.com/drive/folders/1vDhOoXhMTrZaepK7Ai4rG7Xy23BoR_gV?usp=sharing)  and [Best Model](https://drive.google.com/file/d/1ER_K0142FKconjGEXs31KXTMTrNpcCe9/view?usp=sharing) and put them in root folder. 
- Run python train.py to train and python predict.py to test. The best model will be saved on root folder as best_unet_model.pth
  
### Folder Structure
![App Screenshot](./assets/structure.png)

[Source Code](https://github.com/milesial/Pytorch-UNet)




