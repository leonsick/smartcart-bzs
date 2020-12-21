## How to train an SSD Mobile Net V2 Quant8 with Tensorflow

Requirement: Have Tensorflow Object Detection API cloned and setup according to this tutorial up until step 5: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

#### Step One: Setup Google Cloud Deep Learning VM
1. Go to cloud.google.com
2. Login with your Google Account
3. In the search bar, type in "Deep Learning VM" and you will find a pre-defined image for Compute Engine
4. Click on the found "Deep Learning VM" image and start a new instance.
5. In the configuration interface, select us-west-1b as your region and Tensorflow 1.15 as well as the NVIDIA driver 
installed checkbox
6. Start the setup and wait for it to complete
7. Once a predefined terminal command to ssh into the vm appears, copy it and paste it in your terminal.

#### Step Two: Transfer Tensorflow to your VM
1. Zip your prepared Tensorflow folder
2. Upload your .zip file to your Google Drive
3. Right-click on the file and select "show link"
4. There, allow for anyone with the link to see the file i.e. make it public
5. Copy the id from the URL to the file
6. Now go back to your vm. There, install gdown with "pip3 install gdown"
5. Once installed, type in gdown --id <YOUR ID> and paste in your id
6. Your zip will now download to the vm. Once done, unzip it with "unzip <YOUR ZIP FILE>"

#### Step Three: Setup training
1. First, go into the "research" folder and type in: "python3 setup.py build" and then "python3 setup.py install"
2. Next, install some requirements: "pip3 install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tf_slim pycocotools lvis"
3. Then, go the your object_detection/training folder and make sure the paths in the .config file match the ones on your vm.
Note: Your VM path with most probably be "home/<USERNAME>/Tensorflow/..."
4. Set your Pythonpath by typing "export PYTHONPATH=/home/leonsick/Tensorflow/models/research/slim"

#### Step Four: Start training
1. Now your are fully setup! Start training with the following command \n
python3 model_main.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config

Now, wait until your loss consistently drops below 2. 

