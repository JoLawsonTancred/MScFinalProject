# MScFinalProject

For the final project of my MSc in Data Science, I worked in collaboration with the Visualogical art-science collective and the Museum of the Home on the Happy Place game, an interactive exhibition that produced generative art or ‘memes’ in response to player input. Players selected their preferred ‘meme’ out of four options over several generations to produce the meme that best reflected their 'happy place'. My project used deep reinforcement learning to predict and generate a meme that the player would select over randomly generated memes. The theoretical framework is Richard Dawkins' Weasel Progam, which demonstrates the role of selection as well as random mutation in evolution. 

<img width="1156" alt="Screenshot 2022-03-28 at 15 01 26" src="https://user-images.githubusercontent.com/8386425/160414891-9f616c19-da56-45da-8045-e2eae2e106b6.png">

Below is a diagram showing a high-level representation of how deep reinforcement learning can be applied to my research problem. 

<img width="1037" alt="Screenshot 2022-03-28 at 15 20 57" src="https://user-images.githubusercontent.com/8386425/160418908-0ef3d8b7-d792-4d59-9f0e-ea6008ed4e2e.png">

Accuracy is evaluated according to the proximity of the predicted meme to the selected meme in the n dimensional meme space. If this improves with training, I can infer that over successive generations the algorithm is able to learn the players' preferences when selecting a meme. 

I simluated the player's selections in the live game using a custom environment created using OpenAI Gym. I designed the optimum reward function and the 'action' and 'observation' spaces to train the algorithm. I focused on actor-critic reinforcement methods because these are better suited to a high-dimensional environment. 

<img width="853" alt="Screenshot 2022-03-28 at 15 24 20" src="https://user-images.githubusercontent.com/8386425/160419686-e43e5e17-5783-4971-801a-3f8d90760f0d.png">


After studying various state-of-the-art methods, I experimented with and implemented the algorithms A2C, PPO and TRPO using Stable Baselines. Reinforcement learning suffers from sample inefficiency, so I increased my training budget to allow adequate time for the algorithm to converge. Here is an example of how the mean reward increased over 25,000 timesteps while training the A2C alogirthm with the default MLP policy network (gamma=0). 

<img width="967" alt="Screenshot 2022-03-28 at 15 32 20" src="https://user-images.githubusercontent.com/8386425/160421464-9dceb61c-02c5-4f37-8606-be588895cc71.png">
