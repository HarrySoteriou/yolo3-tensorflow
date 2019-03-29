This version of yolo v3 runs using Tensorflow

I got all the code from https://github.com/aloyschen/tensorflow-yolo3 and made some changes in the file config and detect

Along with the scripts you will need to download the following weights file 'YOLOv3-416' from this website https://pjreddie.com/darknet/yolo/ 
and store it inside the model_data folder

Make sure you have downloaded tensorflow packages (pip install tensorflow) and anything else you might need to run the program

Make changes to config file to use your working directory!!!

If you have problems with the Interpreter when using PyCharm for example try to set the project interpreter to the System one

TO DO:
1) Output detects way to few objects, figure out why
2) Encorporate multithreating
3) Download packages that allow for the use of GPU
