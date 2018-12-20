function plotLogResult(xss, x_t, algorithm)
% plot the distance between xss and x_t, xss[-1]
    k = 1:size(xss, 2);
    dis_t = sum((xss - x_t).^2);
    dis_c = sum((xss - xss(:,size(xss, 2))).^2);
    plot(k, log(dis_t), k, log(dis_c));
    title(algorithm);
    xlabel 'iteration k';
    ylabel 'log distance';
    legend("2log||x-x_t||_2", "2log||x-x_c||_2")
end
