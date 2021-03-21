# Banana Agent
The banana agent uses Deep Q-Learning to traverse through a world of bananas.

This project includes scripts that you can use to train and test the agent.

See `Report.md` for explanation of the agent's architecture and training result.

# Environment
The agent is placed in a square world where yellow and blue bananas are randomly placed. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic and is considered solved when the agent gets an average score of +13 over 100 consecutive episodes.

# Requirements
- conda
- [drlnd](https://github.com/udacity/deep-reinforcement-learning#dependencies) environment
- (_For Linux and Mac OSX users_) Unity environment
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

For Linux and Mac OSX users, you will need to place the unity environment file in the current folder and unzip/decompress the file. Afterwards, you will need to navigate to `model.py` and `train.py`, then change the `file_name` parameter for the instantiated `UnityEnvironment` class to match the location of the Unity environment that you downloaded.

# Instructions
### Training
```bash
python train.py
```
Note that once the average score reaches +13 or above, the script will automatically save and update the model weights to `checkpoint.pth` file.

### Testing
```bash
python test.py
```
Note that the default file `saved_model_weights.pth` is loaded. Change this file to `checkpoint.pth` if you want to test an agent that has been successfully trained.
