# Inpainting
Noisy image repair and part missing image repair  

### Description de réparer du bruit d'image:
Nous utiliserons le modèle **LASSO** pour réparer le bruit sur l'image, lire l'image, afficher l'image, ajouter du bruit et d'autres fonctions sur le fonctionnement de base de l'image peuvent être trouvées dans le fichier **img_tools.py**, construire le modèle, le patch et d'autres fonctions peuvent être trouvé dans le fichier **patch_predict.py**. La réparation de l'image complète est implémentée dans le fichier **noise_repair.py**

### Résultat:
![image](https://github.com/Oitron/Inpainting/blob/master/report/repair_noise_res.png)
![image](https://github.com/Oitron/Inpainting/blob/master/report/repair_noise_eval.png)

### Description de réparer des parties manquantes de l'image:
La méthode de complétion que nous avons utilisée est illustrée dans la figure ci-dessus. Les parties manquantes sont parcourues en spirale dans le sens des aiguilles d'une montre et les parties avec une longueur de **step** sont complétées séquentiellement (c'est-à-dire que l'ordre de complétion est indiqué dans le numéro de séquence de la figure ci-dessus). Les méthods d'implémenté peuvent se trouvé dans **mpart_repair.py**.

### Résultat:
![image](https://github.com/Oitron/Inpainting/blob/master/report/original_img.png)
![image](https://github.com/Oitron/Inpainting/blob/master/report/part_missing_img.png)
![image](https://github.com/Oitron/Inpainting/blob/master/report/repair_result.png)
  
![image](https://github.com/Oitron/Inpainting/blob/master/report/original_img2.png)
![image](https://github.com/Oitron/Inpainting/blob/master/report/part_missing_img2.png)
![image](https://github.com/Oitron/Inpainting/blob/master/report/repair_result2.png)


### Source:  
[1] Bin Shen and Wei Hu and Zhang, Yimin and Zhang, Yu-Jin, Image Inpainting via Sparse Representation Proceedings of the 2009 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP ’09)  
[2] Julien Mairal Sparse coding and Dictionnary Learning for Image Analysis INRIA Visual Recognition and Machine Learning Summer School, 2010  
[3] A. Criminisi, P. Perez, K. Toyama Region Filling and Object Removal by Exemplar-Based Image Inpainting IEEE Transaction on Image Processing (Vol 13-9), 2004  
