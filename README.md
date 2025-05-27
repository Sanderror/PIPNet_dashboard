## This file contains the description and guide to our github. 

The first segment of our github is the model itself, forked from the PIPNET model provided to use for this course. To some files in these folders, slight changes have been made in order to allow for inference on user uploaded images. These are the following folders and files:
- features
- pipnet
- used_arguments
- util:
  - data.py: a load_image() function has been added in order to load user uploaded images into test loaders
  - visualize_prediction.py: slight adjustments have been made to the vis_pred() function in order to provide us with predictions on all classes instead of top 3
- main.py:
  - Similar to the main.py of the original PIPNet GitHub, but changes have been made to allow inference on user uploaded images
- nauta_pipnet_cpvr.png

Then there are the following files that we used to create our eventual dashboard/explanation application. The final_app.py file is our final version of the dashboard. More detailed instructions on how to run it can be found in the model_run_instructions.md
- dashboard_images: contains images that are presented in the dashboard
- final_app.py: contains the final version of our dashboard

And finally the instructions on how to run the model, used for ourselves is stored in:

- the model_run_instructions.md 

