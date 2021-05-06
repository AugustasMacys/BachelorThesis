# Deepfakes Detection on the DeepFake Detection Challenge Dataset

If you have any questions, please email me.

## Prerequisites

I add a requirements.txt file to install the libraries.
To install pytorch-coviar, please follow https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md<br/>
To install ffmpeg, please follow https://www.ffmpeg.org/download.html

## SHA256 File Checksum

I add links to OneDrive for dataframes and models because they are large in size.

da3ab25f7c14bbdf91997e25b225553582705f117378d28221db5ababa9a4fbe *./dataframes/coviar_dataframes/coviar_dataframe.csv
0bfabb5782a8f6b306662d80fbb6c9b9fedd38e422afcb7b5bc8fe36f762ffea *./dataframes/coviar_dataframes/coviar_test_dataframe.csv
d9005b09694d9e9ae4e4db33cfa196a024de22377cbe7f84fe5bcefe4e8a4519 *./dataframes/dataframes_3dcnn/3dcnn_dataframe_final.csv
353872d9663215ee877b0c5b37ec83635c7b7ad9b3123ab77c1201459a6f76be *./dataframes/dataframes_3dcnn/3dtesting_dataframe_final.csv
d3a225d71c6855e09fc1eefa949e4f25cbff0099b881437209f14bb5bd55d317 *./dataframes/dataframes_pairs_updated/fake_crops.csv
4c0d27e760a5742233d8c95597d88a28dd3578543659f27167d22e922095bbaa *./dataframes/dataframes_pairs_updated/real_crops.csv
f5aa5f357cb86d2bb8cf40798745cfc0c2bddb1c5ba2c2f6d63985fbb0a620aa *./dataframes/faces_validation_new.csv
8abbb2a7cd0fb9e840e59442efceec52e4eb43445205c02d68e12501e2f33467 *./dataframes/full_facexray_df.csv
fbb0f45b3d9d89c82fc85c55b76b7bed794f28be1f236c859a5cea4667489a19 *./dataframes/testing/labels.csv
e94bccf0fc6fa760c7453e8b4f7bd2d5cc3cb9d128b34a456c25e9f5b3eefd2e *./dataframes/validation/labels_updated.csv
0a3f4582c2c49e35826ea8ec25c472f0baee35bf641b2bf44adb8c4afd8c4a9a *./trained_models/2D_Frozen.pth
b4be5c10f4032d2a0aeb6fa95208e068fe7d5160eeabf1d13b383a00956f10a8 *./trained_models/2D_NotFrozen.pth
a12ac8da8d8d6940f30232eeb9a8c17b8d971ae6fef3525509d63d2f6c74113f *./trained_models/3Dmodel.pth
8b88b07e62a47ca5852f2663289a9f0c3dbc9af82dfb2e7be65615418916db70 *./trained_models/coviar_iframe_model.pth
87f7c44fccf4cc5efc7372eb3faaf43b602e5b1ad3344dbbb44dd9e164cfe71a *./trained_models/coviar_model_motion_vector.pth
e1ce094b4b3d022444f7cce313f82c3f745e5a800861486eb4bbe9bc724daaca *./trained_models/coviar_residual_model.pth
3a04b43eb793c6e029d04fee6fe4bb5fb028a0c2ed8b5ecb88fbc0b6318bbd18 *./trained_models/preview_x_ray_model_B.pth
f49efb71e38cdcbb3d4b523566cc244411be7201b627c29ff99afa5e7ab956ca *./trained_models/preview_x_ray_model_C.pth
60c905581ab5b6d72c267214b7632221013942c7a2e03079b4d82e7abdee1bed *./trained_models/x_ray_model_B.pth
005d7b704dc1845ec947a06762a666944e139964aec3acbafff65b141d07b57a *./trained_models/x_ray_model_C.pth

Link to dataframes:  https://bham-my.sharepoint.com/personal/axm1415_student_bham_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=Rn9%2FBPHJIc9vpA0MIMSt5p556s2WaDrfbbROnIrMciY%3D&folderid=2_04e6b110af948472e9d4bca1869bc865b&rev=1&e=dVmk7O<br/>
Link to models: https://bham-my.sharepoint.com/personal/axm1415_student_bham_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=eyRARd%2Bv6IDxEn4swejtS9Jof0TaBE3YnSoIXZNKQvY%3D&folderid=2_04d869ff4638440bdb4a7db6793a96c9d&rev=1&e=MOVbOf

## General

The code was splitted between notebooks and actual .py files. Notebooks were used to implement specific ideas for small amount of images to test if the ideas work. .py files on another hand, were used to make it work with the full dataset.

## Download the data
We do not include the Dataset because it is very big. Without the data extracted the below instructions will not work except for "Combination Results"

In order to download the data you need to visit the link below:

| Dataset | Links |
| ------ | ------ |
| Training | [https://www.kaggle.com/c/deepfake-detection-challenge/data][PlDb] |
| Validation and Testing | [https://dfdc.ai/login][PlGh] |

Also, it is possible to download the entire dataset with the notebook "DataDownloader" if you load the cookies of your kaggle account.


## Prepare the data

  - Extract faces inside videos with src/video_processing/VideoReader.py by providing the folder of videos and where do you want faces to be extracted
  - Create Dataframes to train based on the faces extracted. This is complicated. We include the dataframes, so that you would not have to do it. Dataframes were created with jupyter notebooks throughout the project.

  
## Training

There are 4 training files you can run:

  - src/training/TrainModelFaces2D.py
  - src/training/TrainModelFaces3D.py
  - src/training_xray/TrainingXRay.py
  - src/training_coviar/CoviarTrainModel.py
  
In order to run it successfully extracted validation faces need to be placed under "validation_faces" folder and specific dataframes need to be placed under "dataframes" folder.

## Testing

In order to test on the testing dataset you need to run:

  - src/inference/Inference.py for 2D and 3D models
  - src/training_coviar/CoviarCombination.py for compressed video classifier
  - src/training_xray/TrainingXRay.py for x ray models. Uncomment the testing code provided.


## Combination Results

We include 3 prediction files which were used to get predictions. You can run the notebook file:

notebooks/ResultsAnalyser.ipynb



