function plotResult(xss, x_t, algorithm)
% plot the distance between xss and x_t, xss[-1]
    k = 1:size(xss, 2);
    dis_t = sum((xss - x_t).^2);
    dis_c = sum((xss - xss(:,size(xss, 2))).^2);
    plot(k, dis_t, k, dis_c);
    title(algorithm);
    xlabel 'iteration k';
    ylabel 'log distance';
    legend("||x-x_t||_2", "||x-x_c||_2")
end
