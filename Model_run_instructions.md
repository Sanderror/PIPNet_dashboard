# Instructions to run the app
## Step 1: Clone the project

## Step 2: Install packages
* Ensure that you have all the required packages installed to run the code
* Do not forget streamlit

## Step 3: Add data folder
* From this link: https://drive.google.com/file/d/1PfeW5afu3cSdWTi03Ac3PYH8O1BSp9kK/view you can download the data, and place that in the project directory under the folder name 'data'
* Inside that same data folder, **add another folder called 'user_images'**, and inside that folder place a folder called **'uploaded_images'** in which the user uploaded images will be stored

## Step 4: Add the pretrained network and visualized_prototypes folder
* From this other link: https://drive.google.com/file/d/195oPh4-ugl8LkqFrlPwzHaZ3rH16Jc7a/view you can download the pretrained network (and some other files like the visualized prototypes). Place all the contents of this zip inside your project
* Most important are the **checkpoints folder** which includes the net_trained (the pretrained network), and the **visualized_prototypes** folder, which will be used in the dashboard
* The other folders are not necessary.

## Step 5: Launch the app
* You can launch the app by typing in the terminal (in your project directory, in your virtual environment): **streamlit run final_app.py**

## Step 6: Upload images you want to get predicted
* The app will be launched in your browser, and there you can upload the images that you would like to be predicted
* There are also other pages in the app explaining everything you need to know about the dashboard

## Step 7: Predict
* Click on the 'Predict' button in the app, and wait for the model to run (around 30-40 seconds if you just uploaded a few images)
* Then, the dashboard will provide 3 predictions for the uploaded image with detailed explanations using PIPNet's patch to prototype framework
