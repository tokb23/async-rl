# Asynchronous Methods for Deep Reinforcement Learning in TensorFlow + OpenAI Gym
This is an implementation of Asynchronous Methods for Deep Reinforcement Learning (based on [Mnih et al., 2016](https://arxiv.org/abs/1602.01783)) in TensorFlow + OpenAI Gym.  

## Requirements
- gym (Atari environment)
- scikit-image
- tensorflow

## Results
Coming soon...

## Usage
#### Training
For asynchronous advantage actor-critic, run:

```
python main.py
```

#### Visualizing learning with TensorBoard
Run the following:

```
tensorboard --logdir=summary/
```

## References
- [Mnih et al., 2016, Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [coreylynch/async-rl](https://github.com/coreylynch/async-rl)
- [miyosuda/async_deep_reinforce](https://github.com/miyosuda/async_deep_reinforce)
- [muupan/async-rl Wiki](https://github.com/muupan/async-rl/wiki)
- [stackoverflow: Asynchronous computation in tensorflow](http://stackoverflow.com/questions/34419645/asynchronous-computation-in-tensorflow)
