function result = sumBcoeff(m, n, theta, phi, Etheta)
    Mtheta = length(theta);
    Nphi = length(phi);
    result = zeros(Mtheta,Nphi);
    for mtheta = 1:Mtheta
        if theta(mtheta) ~= 0 && theta(mtheta) ~= pi
            legendre_cos = legendre(n,cos(theta(mtheta)));
            legendre_cos_mas = legendre(1 + n,cos(theta(mtheta)));
            
            legendre_deriv = -(1 + n)*cot(theta(mtheta))*legendre_cos(m+1) + ...
            (1 + n - abs(m)) * csc(theta(mtheta))* legendre_cos_mas(m+1)
            
            valor_derivada= legendre_deriv;

        elseif m == 1
                valor_derivada= -(1/2)*n*(1 + n);
        else 
                valor_derivada= 0;
        end
        for nphi = 1:Nphi
            result(mtheta,nphi) = (Etheta(mtheta) * sin(theta(mtheta)) * ...
                valor_derivada) * exp(-1i * m * phi(nphi));
        end
    end
end