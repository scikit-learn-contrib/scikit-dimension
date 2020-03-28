# TWO-NN
Intrinsic dimension estimation tools

Usage: 

./TWO-NN.x -input <filename> {-coord|-dist} [-discard <fraction>]


Files:

In the case of a file of coordinates, every column corresponds to a coordinate ( and each row represents a point)
In the case of distances, an upper diagonal matrix has to be passed to the program, in the form index1 >> index2 >> dist(index1, index2);

Outputs:

r_list.dat file containing the list of distances between the point and its first and second neighbor
nu_list.dat file containing the list of nu values   
fun.dat file containing the coordinates to plot the S-set

The program produces also two plot, the block analysis plot and the S-set plot

