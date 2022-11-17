%image



function Inew = orthoimages_with_dim(dim)

I = ones(round(dim/4));


x1=[I -I;
    -I  -I];

A1= [x1 -x1;
    -x1  -x1];
B1= [-x1 x1;
    -x1  -x1];
C1= [-x1 -x1;
    x1  -x1];
D1= [-x1 -x1;
    -x1  x1];

x2=[-I I;
    -I -I];


A2= [x2 -x2;
    -x2  -x2;];
B2= [-x2 x2;
    -x2  -x2;];
C2= [-x2 -x2;
    x2  -x2;];
D2= [-x2 -x2;
    -x2  x2;];

x3=[-I -I;
     I  -I];


A3= [x3 -x3;
    -x3  -x3;];
B3= [-x3 x3;
    -x3  -x3;];
C3= [-x3 -x3;
    x3  -x3;];
D3= [-x3 -x3;
    -x3  x3;];


x4=[-I -I;
   -I I];

A4= [x4 -x4;
    -x4  -x4;];
B4= [-x4 x4;
    -x4  -x4;];
C4= [-x4 -x4;
    x4  -x4;];
D4= [-x4 -x4;
    -x4  x4;];

Inew(:,:,1) = A1;
Inew(:,:,2) = A2;
Inew(:,:,3) = A3;
Inew(:,:,4) = A4;
Inew(:,:,5) = B1;
Inew(:,:,6) = B2;
Inew(:,:,7) = B3;
Inew(:,:,8) = B4;
Inew(:,:,9) = C1;
Inew(:,:,10) = C2;
Inew(:,:,11) = C3;
Inew(:,:,12) = C4;
Inew(:,:,13) = D1;
Inew(:,:,14) = D2;
Inew(:,:,15) = D3;
Inew(:,:,16) = D4;