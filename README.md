### Requirements:
- for running the program on the GPU (CUDA support) you need to build opencv from source with cmake (https://www.youtube.com/watch?v=tjXkW0-4gME)
- otherwise just `pip install -r requirements.txt`


### How to use:
- run `getModels.sh` to download the predefined models
- after all requirements are installed, there are many programs that can be run
1. `take_a_video.py` this allowes the user to make a video to be processed later
2. `biceps_main.py` in which you can specify which video to process for the biceps exercise
3. `squat_main.py` in which you can specify which video to process for the squat exercise
4. `Camera2.py` where you have steps 1. and 3. combined, so, it takes the video, after you finish, its processes the video and gives you the output for the squat exercise
- all videos will be saved in the `videos` folder




