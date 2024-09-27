function result = SphericalHankelH2(n,z)
%SPHERICALHANKELH2 calcula la función de hankel esférica de orden 2
    sphbesselj = ((-1)^(n+1)) * sqrt(pi/(2*z))*besselj(-n-0.5, z);
    sphbessely = ((-1)^(n+1)) * sqrt(pi/(2*z))*bessely(-n-0.5, z);
    h2 =  -sphbessely  - i*sphbesselj;
    result = h2;
end

