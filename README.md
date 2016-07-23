# Asynchronous Methods for Deep Reinforcement Learning in Keras + TensorFlow + OpenAI Gym
This is an implementation of Asynchronous Methods for Deep Reinforcement Learning (based on [Mnih et al., 2016](https://arxiv.org/abs/1602.01783)) in Keras + TensorFlow + OpenAI Gym.  

## Requirements
- gym (Atari environment)
- scikit-image
- keras
- tensorflow

## Results
Coming soon...

## Usage
#### Training
For asynchronous advantage actor-critic, run:

```
python a3c.py
```

#### Visualizing learning with TensorBoard
Run the following:

```
tensorboard --logdir=summary/
```

## References
- [Mnih et al., 2016, Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [coreylynch/async-rl](https://github.com/coreylynch/async-rl)
