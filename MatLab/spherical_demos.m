clc;clear;close all


f = 1e9;
lambda = 3e9/f;
N = floor(2*pi*lambda);
%Angulos
puntos_del_campo = 50;
theta = linspace(0, pi, puntos_del_campo);
phi = linspace(0, 2*pi, puntos_del_campo);

%Variables campo
etha = 120/pi;
k = 2*pi/lambda;
Io=1;
l=1;
r = 1;
%Campo
E_theta = i*etha*((k*Io*l*sin(theta))/(4*pi*r))*(1+ (-i/k*r) - (1/(k*r)^2))*exp(-i*k*r);
E_phi = 0;
