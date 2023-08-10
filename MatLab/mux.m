clc;clear;close all

%--------------------------------------------------------------------------
%Datos
f = 1e9;
lambda = 3e9/f;
N = floor(2*pi*lambda);
%Angulos
puntos_del_campo = 50;
theta = linspace(0, pi, puntos_del_campo);
phi   = linspace(0, 2*pi, puntos_del_campo);
z = 1
%--------------------------------------------------------------------------
%Variables campo
etha = 120/pi;
k = 2*pi/lambda;
Io=1;
l=1;
r = 1;
%Campo
E_theta = i*etha*((k*Io*l*sin(theta))/(4*pi*r))*(1+ (-i/k*r) - (1/(k*r)^2))*exp(-i*k*r);
E_phi = 0;
%--------------------------------------------------------------------------
%Polinomio asociado de legendre

legendre_cos = legendre(1,cos(theta));
legendre_cos = legendre_cos(2,:);

legendre_cos_mas = legendre(1 + n,cos(theta));
legendre_cos_mas = legendre_cos_mas(2,:);

legendre_deriv = -(1 + n)*cot(theta).*legendre_cos + ...
            (1 + n - abs(m)) * csc(theta) .* legendre_cos_mas;

%--------------------------------------------------------------------------
%Función esférica de henkel de orden 2
sphbesselj = ((-1)^(n+1)) * sqrt(pi/(2*z))*besselj(-n-0.5, z);
sphbessely = ((-1)^(n+1)) * sqrt(pi/(2*z))*bessely(-n-0.5, z);
h2 =  -sphbessely  - i*sphbesselj;

%--------------------------------------------------------------------------
%Cálculo de Amn

% Amn = zeros(1,puntos_del_campo)


Amn_integral = @(tetha,phi) (E_theta*m*legendre_cos)*exp(-i*m*phi)
integral2(fun,0,1,2,3)





