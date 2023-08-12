clc;clear;close all

%--------------------------------------------------------------------------
%Datos
f = 1e9;
lambda = 3e8/f;
N = floor(2*pi*lambda);
n = 1;
m = 1;
%Angulos
puntos_del_campo = 11;
theta = linspace(0, pi, puntos_del_campo);
phi   = linspace(-pi, pi, puntos_del_campo);

delta_theta = theta(2)-theta(1);
delta_phi   = phi(2)-phi(1);
%--------------------------------------------------------------------------
%Variables campo
etha = 120/pi;k = 2*pi/lambda;Io=1;l=1;r = 1;z = k*r;
%Campo
E_theta = i*etha*((k*Io*l*sin(theta))/(4*pi*r))*(1 +(1/(i*k*r)) - (1/(k*r*k*r)))*exp(-i*k*r);
E_theta = theta;
E_phi = zeros(1,puntos_del_campo);
%--------------------------------------------------------------------------
%Polinomio asociado de legendre
legendre_cos = legendre(n,cos(theta));
legendre_cos = legendre_cos(1 + n,:);

legendre_cos_mas = legendre(1 + n,cos(theta));
legendre_cos_mas = legendre_cos_mas(1 + n,:);

legendre_deriv = -(1 + n)*cot(theta).*legendre_cos + ...
            (1 + n - abs(m)) * csc(theta) .* legendre_cos_mas;

%--------------------------------------------------------------------------
%Función esférica de henkel de orden 2
sphbesselj = ((-1)^(n+1)) * sqrt(pi/(2*z))*besselj(-n-0.5, z);
sphbessely = ((-1)^(n+1)) * sqrt(pi/(2*z))*bessely(-n-0.5, z);
h2 =  -sphbessely  - i*sphbesselj;

%--------------------------------------------------------------------------
%Cálculo de Amn

Amn_parte1 = -i*(2*n+1)/(4*pi*h2)*(factorial(n-m)/factorial(n+m))*(1/(n*(n+1)))*delta_phi*delta_theta;

Amn_parte2 = sumAcoeff(abs(m), n, theta, phi, E_theta);
Amn = sum(Amn_parte1*sum(Amn_parte2))
% -0.00678276 - 0.00147836i

%--------------------------------------------------------------------------
%Cálculo de Bmn

Bmn_parte1 = -i*(2*n+1)/(4*pi*h2)*(factorial(n-m)/factorial(n+m))*(1/(n*(n+1)))*delta_phi*delta_theta;

Bmn_parte2 = sumAcoeff(abs(m), n, theta, phi, E_theta);
Bmn = sum(Bmn_parte1*sum(Bmn_parte2))
% -0.00678276 - 0.00147836i




