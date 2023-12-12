function I = peopleid2pp(keyp, joint, keypadini,keypchini,i)

        distad = sqrt(sum((keyp(:,28,i)-keypadini).^2));
        distch = sqrt(sum((keyp(:,28,i)-keypchini).^2));
        distances = [distad, distch];
        [M,I] = min(distances);
end
