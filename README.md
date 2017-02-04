# README #

### Initialization steps

You have to a system with `pyhton3` and `pip`. I recommend install `virtualenv` to set an environment on your project folder, then, with command `install -r requirements.txt`, all dependencies will be installed.
For some graph `pyhton3-tk` is required so I need to install it with `apt-get install pyhton3-tk`


#### TensorBoard
You can visualize your network trough your browser. You have to activate your *virtualenv* and run **tensorboard** command
 ```
 ~/$ source bin/activate
 (ProjectName) ~/$  python -m tensorflow.tensorboard --logdir=/path/to/board/folder
 ```