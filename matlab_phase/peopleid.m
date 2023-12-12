function I = peopleid(keyp, joint, keypadini,keypchini, keypoini,i)

        distad = sqrt(sum((keyp(:,28,i)-keypadini).^2));
        disto = sqrt(sum((keyp(:,28,i)-keypoini).^2));
        distch = sqrt(sum((keyp(:,28,i)-keypchini).^2));
        distances = [distad, distch, disto];

        [M,I] = min(distances);
end
