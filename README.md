# Group GMM-ResNet for Detection of Synthetic Speech Attacks
# Zhenchun Lei, Yan Wen, Yingen Yang, Changhong Liu, Minglei Ma
# zhenchun.lei@hotmail.com

Step 1: run as21_feature.py
Extracting all LFCC features.

Step 2: run matlab/asvspoof21_ms_gmm_augment.m
Training the GMMs.

Step 3: run as21_gmm_group_resnet_exp.py
The Group GMM-ResNet model.


@inproceedings{lei23_interspeech,
  author={Zhenchun Lei and Yan Wen and Yingen Yang and Changhong Liu and Minglei Ma},
  title={{Group GMM-ResNet for Detection of Synthetic Speech Attacks}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={3187--3191},
  doi={10.21437/Interspeech.2023-1249}
}
