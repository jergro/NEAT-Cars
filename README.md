# Self driving cars using Neuroevolution of augmenting topologies

### Usage
Play the game: **car_game.py**
Train the algorithm:  **car_train.py**

Some packages need to be installed. Mainly pygame and neat.

### Environment
The environment is created using Pygame. It is available to try out in **car_game.py**. 

The map is drawn by hand. A mask of the track is obtained where the alpha-values change in the png.
Wall detection is done through ray-casting and intersection with the mask, it is not optimized and therefore quite slow for many rays. 
Although it works splendid for one car.

### NEAT-Algorithm
The NEAT package is used to implement the neat algorithm. A configuration file is included in the repository.
It is a fully connected feed-forward network. The Topology of the network is changing as the network is training. 

## NOTES:
#### Refactoring:
* Rewrite functions, each function should do one thing and one thing only
* Optimize ray-casting, today it is very heavy computationaly

### To Add:
* Add game-menu in order to chose free play or train the algo
* Add a sandbox mode where the terrain can be drawn.


Sidenote: Rewrite in Unity?
