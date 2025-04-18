# Instructions to run the code (no app)
## Step 1: Clone the project
* In this Github, click on the green 'Code' button, copy the HTTPS link, and put it in your python application (Pycharm, VScode etc.)
* I dont know how it works for VScode but in Pycharm you can click on 'Get from VCS' when making a new project, and just paste the link there

## Step 2: Install packages
* Ensure that you have all the required packages installed to run the code

## Step 3: Add data folder
* From this link: https://drive.google.com/file/d/1PfeW5afu3cSdWTi03Ac3PYH8O1BSp9kK/view you can download the data, and place that in the project directory under the folder name 'data'
* Inside that same data folder, add another folder called 'user_images', and inside that folder place a folder called 'uploaded_images' in which you should upload a test image (of a bird) that you want to get classified

## Step 4: Add the pretrained network
* From this other link: https://drive.google.com/file/d/195oPh4-ugl8LkqFrlPwzHaZ3rH16Jc7a/view you can download the pretrained network (and some other files on already visualized prototypes). Place all the contents of this zip inside your project
* Most important is the checkpoints folder which includes the net_trained (the pretrained network)

## Step 5: Add argument/parameter to configuration to let the model use the pretrained network
* I am not sure how to do this in VScode or others, but in Pycharm you can click at the top on the 'Current file' or 'main' or whatever it says. Then you can add a configuration for the main.py file (by just selecting the path to the main.py file), then in the empty field you can add the argument: --state_dict_dir_net net_trained
* ![image](https://github.com/user-attachments/assets/71d5d646-a507-434c-bb98-62dacc84af0c)
* ![image](https://github.com/user-attachments/assets/b48b16b3-c675-497d-8a38-95e8df8a4889)
* ![image](https://github.com/user-attachments/assets/4e84ee73-d3ba-4842-831e-486250709b15)

## Step 6: Run the main.py file and find results
* Simply then run the main.py file, this will take like a minute to run or so
* Then, a folder is created 'runs/run_pipnet' and within that folder you can find the folder 'visualization_results' with the classification results for your uploaded images
* In that folder, for every image there is another folder and within that image folder there are 3 folders for the 3 best predictions of the model for the image
* You can interpret the scores, the prototypes used, etc. and if you want to see what prototype was used, you can use the code p245 for example, and go to the visualised_prototypes folder in the project, and search for prototype_245 to see the prototype used.

# Instruction to run the app
## Step 1: Install streamlit package into your environment

## Step 2: Launch the app
* You can launch the app by typing in the terminal (in your project directory, in your environment): streamlit run app.py

## Step 3: Upload test images
* The app will be launched in your browser, and there you can upload the images that you would like to be predicted

## Step 4: Predict
* Click on the 'Predict' button in the app, and wait for the model to run (around 40 seconds if you just uploaded a few images)
* The dashboard will display the names of the uploaded images and below that the top 3 predicted classes with their scores
