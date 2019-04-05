This version of yolo v3 runs using Tensorflow

I got all the code from https://github.com/aloyschen/tensorflow-yolo3 and made some changes in the file config and detect

Along with the scripts you will need to download the following weights file 'YOLOv3-416' from this website https://pjreddie.com/darknet/yolo/ 
and store it inside the model_data folder

Make sure you have downloaded tensorflow packages (pip install tensorflow) and anything else you might need to run the program

Make changes to config file to use your working directory!!!

If you have problems with the Interpreter when using PyCharm for example try to set the project interpreter to the System one

TO DO:
1) Takes 20-30 sec to detect objects for each picture which is too long and I don't know why.
2) Encorporate multithreating
3) Download packages that allow for the use of GPU

ANSWERS:
1) Yolo demands a lot of processing power and appears to operate much faster on Ubuntu (11 sec/ picture)
2) We have checked the CPU usage and all cores were used to the max. Maybe parallelisation is possible but this is unknown for the moment
3) Will be working on this today (05/04)

TO DO:
1) Run the code on PySpark
2) Test yolo from a different source https://github.com/AlexeyAB/darknet
3) Try to preprocess the images before feeding them into the code
4) Report on improvement when using GPU https://medium.com/@lisulimowicz/tensorflow-cpus-and-gpus-configuration-9c223436d4ef

