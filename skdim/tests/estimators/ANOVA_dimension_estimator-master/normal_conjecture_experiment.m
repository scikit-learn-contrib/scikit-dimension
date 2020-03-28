%% Experiment testing Gaussian limit
%   This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%
%   Be aware this experiment might take a couple of hours.
clc; close all; clear all;
D = [2 10 20 30 40 50 60 70 80 90 100];
M = length(D);
betas = compute_betas(max(D));
acceptance_ratio = zeros(M, 1);
p_values = zeros(M,1);
N = 50;
for jj = 1:M
    d = D(jj);
    fprintf('Testing dimension %d \n', d);
    for ii = 1:N
        U = simulate_statistic(d,10*d,betas(d), 2500);
        [h, p] = adtest(U, 'Alpha', 0.025);
        p_values(jj) = p_values(jj) + p;
        if h == 0
            acceptance_ratio(jj) = acceptance_ratio(jj) + 1;
        end
    end
end
acceptance_ratio = acceptance_ratio/N;
p_values = p_values/N;
plot(acceptance_ratio);
%% Acceptance ratio plot
% Creates a fancy figure and saves it
% In order to use this code you need to download linspecer
% https://github.com/davidkun/linspecer and export_fig https://github.com/altmany/export_fig

close all;
Cc = linspecer(4); 
plot(D, acceptance_ratio, 'Color', Cc(1,:), 'Linewidth',2.5);
set(gca,'fontsize',18)
ylabel('Ratio acceptance/#trials');
xlabel('Dimension');
export_fig('results/acceptance_ratio', '-r300', '-png', '-transparent');
%% Box plot experiment
D = [2 25 75 100];
M = length(D);
betas = compute_betas(max(D));
acceptance_ratio = zeros(M, 1);
N = 50;
p_values = zeros(M,N);
for jj = 1:length(D)
    d = D(jj);
    fprintf('Testing dimension %d \n', d);    
    for ii = 1:N
        U = simulate_statistic(d,10*d,betas(d), 2500);
        [h, p] = adtest(U, 'Alpha', 0.025);
        p_values(jj,ii) = p;
        if h == 0
            acceptance_ratio(jj) = acceptance_ratio(jj) + 1;
        end
    end
end
acceptance_ratio = acceptance_ratio/N;
p_values = p_values;
boxplot(p_values');
%% Fancy box plots
% Creates a fancy Figure and saves it
% In order to use this code you need to download linspecer
% https://github.com/davidkun/linspecer and export_fig https://github.com/altmany/export_fig

close all;
Cc = linspecer(4); 
boxplot(p_values', D);
set(gca,'yscale','log','fontsize',18)
a = get(get(gca,'children'),'children');
xlabel('Dimension');
ylabel('p-value');
for ii = 1:M
    set(a(ii), 'MarkerEdgeColor', Cc(ii,:), 'Linewidth', 2);
    set(a(M + ii), 'Color', Cc(ii,:), 'Linewidth', 2);
    set(a(2*M +ii), 'Color', Cc(ii,:), 'Linewidth', 3.5); 
    set(a(3*M +ii), 'Color', Cc(ii,:), 'Linewidth', 2); 
    set(a(4*M +ii), 'Color', Cc(ii,:), 'Linewidth', 2); 
    set(a(5*M +ii), 'Color', Cc(ii,:), 'Linewidth', 2); 
    set(a(6*M +ii), 'Color', Cc(ii,:), 'Linewidth', 2);
end
export_fig(sprintf('results/boxplot', D(jj)), '-r300', '-png', '-transparent');
%% QQ plots experiment
close all;
D = [2, 25, 50];
M = length(D);
betas = compute_betas(max(D));
N = 1000;
U = zeros(M, N);
for jj = 1:M
    d = D(jj);
    U(jj,:) = simulate_statistic(d,10*d,betas(d), N);
%     Uncomment the following two lines to get a simple qq plot
%     figure; 
%     qqplot(U(jj,:));
end

%% Fancy QQ plots
% Creates a fancy figure and saves it
% In order to use this code you need to download linspecer
% https://github.com/davidkun/linspecer and export_fig https://github.com/altmany/export_fig

close all; clc;
Nc=2;
Cc = linspecer(Nc); 
for jj = 1:length(D)
    close all;
    h = qqplot(U(jj,:));
    set(h(3),'LineStyle','-')
    set(h(3),'LineWidth',2.5);
    set(h(3),'Color',Cc(1,:));
    set(h(1),'LineWidth',2);
    set(h(1),'MarkerEdgeColor',Cc(2,:));
    set(gca,'fontsize',18, 'color', 'w') %set font and tickmark size
    title('')
    ylabel('Quantiles of Sample');
    export_fig(sprintf('results/qqplot2%d', D(jj)), '-r300', '-png', '-transparent');
end