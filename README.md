# physarum-maze
This project is an implementation of a paper by Atsushi Tero, Ryo Kobayashi and Toshiyuki Nakagaki[1].

This current-reinforcement based model, inspired by the Physarum Polycephalum slime mold, lets you solve mazes and find optimal networks.

## Usage
In the notebook you can:
* specify various constants (time step delta, the function f used to calculate the derivative (defaults to Q^mu), number of simulation steps, etc.)
* specify a graph to run the model on by providing the distance matrix
* specify a maze to solve by providing the maze in graphical format, and using the provided helper function to convert it to a distance matrix
Then, the model can be run and, in case you are using a maze, the solution can be plotted.

The python file solves mazes in the same way as the notebook. Running it with `matplotlib` and `numpy` installed will solve the given maze and will display and update the state of the physarum in the maze every ten iterations.

Final note: `tqdm` is used to show the progress of the simulation.

![](https://raw.githubusercontent.com/robert-30/physarum-maze/master/phys.gif)
## References
[1] Atsushi Tero, Ryo Kobayashi, Toshiyuki Nakagaki,
A mathematical model for adaptive transport network in path finding by true slime mold,
Journal of Theoretical Biology,
Volume 244, Issue 4,
2007,
Pages 553-564
