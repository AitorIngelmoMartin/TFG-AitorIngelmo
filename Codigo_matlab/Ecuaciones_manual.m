clear;clc;close all;


fun = @(x) exp(-x.^2).*log(x).^2;
q = integral(fun,0,Inf)

funcion = @(x,y) sin(x.^2)+ cos(y);
salida= integral(funcion,0,inf)