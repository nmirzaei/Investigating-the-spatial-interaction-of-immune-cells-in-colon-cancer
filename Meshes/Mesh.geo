// Gmsh project created on Wed May 16 11:01:09 2018
lc=0.05;
//+
//You have to go to Options > Mesh > Advanced and Uncheck "Extend element sizes from boundary" for the boundary layer to happen
//+
Point(1) = {0, 0, 0,lc};
//+
Point(2) = {0.5, 0, 0};
//+
Point(3) = {-0.5, 0, 0};
//+
Point(4) = {0, 0.5, 0};
//+
Point(5) = {0, -0.5, 0};
//+
Circle(1) = {4, 1, 3};
//+
Circle(2) = {4, 1, 2};
//+
Circle(3) = {2, 1, 5};
//+
Circle(4) = {5, 1, 3};
//+
Line Loop(1) = {1, -4, -3, -2};
//+
Plane Surface(1) = {1};
//+
//+
//Point{1} In Surface{1};
//+
Physical Line(1) = {1, 2, 3, 4};
//+
Physical Surface(2) = {1};
//+
//Uncomment the following lines if you need refinement
//Field[1] = Distance;
//Field[1].EdgesList = {1,2,3,4};
//Field[1].NNodesByEdge = 500;
//+
//Field[2] = Threshold;
//Field[2].IField = 1;
//Field[2].LcMin = lc / 4;
//Field[2].LcMax = lc;
//Field[2].DistMin = 0.04;
//Field[2].DistMax = 0.05;
//+
//Background Field = 2;
