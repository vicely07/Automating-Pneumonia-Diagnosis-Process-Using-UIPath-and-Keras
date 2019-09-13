## Automating-Pneumonia-Diagnosis-Process-Using-UIPath-and-Keras
by Vi Ly
In Participation in UIPath Global Automation Hackthon 2019
## How to use this UIPath component? 
**Requirement:**
•	Windows 10
•	Anaconda
•	Python 3.7
•	Python packages: Keras, CV2, H5PY
## Step by step instructions:
1.	Git clone the repository from: https://github.com/vicely07/Automating-Pneumonia-Diagnosis-Process-Using-UIPath-and-Keras. A folder called “Automating-Pneumonia-Diagnosis-Process-Using-UIPath-and-Keras” will be created in the location of your choice. 

2.	Unzip the folder “Sample Images”.

3.	Download the best_weight folder from my Dropbox into the folder “Automating-Pneumonia-Diagnosis-Process-Using-UIPath-and-Keras”. Download the folder from here: https://www.dropbox.com/sh/l3o8laoit7f2kyq/AACDjQQkZc2Bh3gi7kk3W6HLa?dl=0

4.	Go to the file Predict.py in src folder and change the line “path = 'C:/Users/lykha/OneDrive/Documents/GitHub/Automating-Pneumonia-Diagnosis-Process-Using-UIPath-and-Keras/' “ to your working directory and save the change. Please make sure the format is similar to my original directory.

5.	Open UIPath Studio, load the Main.xaml from the UIPath folder. When the workflow is loaded, go to the sequence called “Output” and find the activity called “Read Text File”. Change the directory in there to your working directory so that the result can locate the result correctly. 

6.	After you do all the above, the workflow is ready to run. Click “Run” to start the processs. A message box will pop up to ask you for password, type in: 12345. 

7.	When a window pop up to ask you for an input image, navigate to the folder “Sample Images” and double click on an image from there. The image will be loaded into the deep learning model.

8.	When a message box pop up asking you if you want to display result, type “yes” (all lower case)

9.	Finally, the result will pop up together with the testing image. A message will pop up to ask if you want to test again or not.

## Inspiration:
Pneumonia is an infection in one or both lungs caused by bacteria, viruses, or fungi. Pneumonia causes inflammation in the air sacs in your lungs which will cause it difficult to breathe. Each year in the United States, about 1 million people have to seek care in a hospital due to pneumonia. In those 1 million, about 50,000 people die from the disease each year in the United States. 

According to National Institutes of Health (NIH), chest X-ray is the best test for pneumonia diagnosis. However, reading x ray images can be tricky and requires domain expertise and experience. With the high rise in popularity of neural networks, researchers and engineers have been able to find many world-changing applications for computer vision. 

Deep learning now allows us to easily create artificial intelligence to help automate analysis techniques which were previously thought impossible for computers. In this project, I use deep learning model to accurately diagnose pneumonia through chest x-ray image inputs and UIPath automating the deep learning training and testing process. 

## How I built it
**Dataset:**
Thanks to Kaggle, I was able to obtain this dataset of over 6000 pneumonia x-ray scans, which already came labeled! There was one folder named “Normal Scans” and another “Pneumonia Scans”. Not all the images were formatted the same way, so I had to uniformly make them all 224x224 pixel RGB images.

**Architectures:**
1. I chose a simple architecture to work on.
2. Because first few layers capture general details like color blobs, patches, edges..., so, instead of randomly initialized weights for these layers, it would be much better if you fine tune them by importing the pretrained weights from imagenet.
3. I added layers that introduce a lesser number of parameters. For example, SeparableConv in Keras is a good replacement for Conv layer. It introduces less number of parameters and filters comparing to normal convolutionwhile capturing more information. 
4. I also added batch norm with convolutions. For a deep network, batch norm is an efficient choice.
5. I put dense layers at the end and trained with a higher learning rate and experiment with the number of neurons in the dense layers. Once the model learnt a good depth, I started training my network with a lower learning rate along with decay. 

**Keras Deep Learning Model:**
![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/821/653/datas/gallery.jpg)


**Metrics:**
Accuracy: 90%;
False positive: 2%;
False Negative: 42%;
Recall: 98%;
Precision: 80%;

**Confusion Matrix Evaluation of the Keras Model:**
![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/821/645/datas/gallery.jpg)

UIPath:
After saving the best deep learning model, I built a workflow in UIPath to automate my testing process. The workflow will load the deep learning model, ask the users to select their chest Xray image and use the pre-trained model to test on the selected image. The diagnosis results will be displayed in a message box along with the selected image. All results will be saved into a log so that users can keep track of their testing process. 

## Challenges I ran into:
I tried to run the Python file directly on the UIPath server but there are many deep learning packages that is not yet available. Therefore, after several trials and failures implementing different UIPath workflow, I eventually arrive to a working process by directing UIPath Robot to run the Python file on terminal while exporing the results to txt file. 

## Accomplishments that I'm proud of
Pneumonia diagnosis itself is not an easy procedure and the availability of the right personnel across the globe is also a serious concern. We looked at easy to build open-source techniques leveraging AI which can give us state-of-the-art accuracy in detecting malaria thus enabling AI for affordable healthcare, especially in developing country. Let’s hope for more adoption of open-source AI capabilities across healthcare making it cheaper and accessible for everyone across the globe!

## What I learned:
I am a newbie in using UIPath Studio so there are so many new information I gained from this projects. I find that UIPath is super easy to get started with and really useful in helping me automate the tedious and repetitive tasks so that I have more time in the interesting work of perfecting my deep learning models.  

## What's next for the Project:
I will implement the UIPath workflow so that it can test the deep learning model on a whole folder of chest Xray images. I will also develop deep learning models for many other medical images such as blood smear, MRI or ultrasound. 
