function result = sumAcoeff(m, n, theta, phi, Etheta)
    Mtheta = length(theta);
    Nphi = length(phi);
    result = 0;
    for mtheta = 1:Mtheta
        legendre_cos = legendre(n,cos(theta(mtheta)));
        for nphi = 1:Nphi
            result(mtheta,nphi) = (Etheta(mtheta) * m * legendre_cos(m+1)) * exp(-1i * m * phi(nphi));
        end
    end
end