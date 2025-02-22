# GPRO-implementation
Attempt at GPRO (policy network no value network) in non LLM scenarios

# GPRO
GPRO is known for its uses in Large Language Learning Models like Deepseek r1. However, while google searching about it, little was shown about its utilisation in other fields, especially because of the time save from disregarding the value network entirely. I experiement around using this technique of reinforcement learning in areas unrelated to LLMs like CartPole.

## How it works (LLM)
In LLMs, the model is given a prompt, and it is tasked to generate multiple responses to the prompt. These responses are then given "advantages" according to the normalised rewards they accrue

### Normalisation function:
```advantage = (rewards - rewards.mean) / (rewards.std() + 1e-5)```
Note we add 1e-5 to prevent zerodivision error

### Post norm
For the responses with positive advantage, during training, the model will be tweaked such that it chooses these tokens more often, and the tokens in the negative advantages will be chosen less often.
Hence, this type of training does not require a value network at all.

## Usage in non LLM scenarios
I implement this is different gym environments. To simulate the process of generating multiple responses, the policy network generate multiple "trajectories" from a single state all the way until the environment terminates. There, rewards are calculated and then normalised, and the training is the same.
