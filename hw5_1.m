function [centroids, labels] = hw5_1
    % Generate random 2d data
    n = 40; % data points per cluster
    means = [10 0; 0 10; 0 0];
    c = size(means,1);
    stdevs = [1 2 3];
    data = zeros(n*c,2);
    true_labels = zeros(n*c,1);
    for cx = 1:c
        data(n*(cx-1)+1:n*cx,1) = normrnd(means(cx,1),stdevs(cx),n,1);
        data(n*(cx-1)+1:n*cx,2) = normrnd(means(cx,2),stdevs(cx),n,1);
        true_labels(n*(cx-1)+1:n*cx) = cx;
    end
    set(gcf,'Position',[250 250 1200, 300]);
    subplot(1,3,1);
    scatter(data(:,1),data(:,2),50,true_labels,'filled','LineWidth',2);
    title("Ground truth");

    [centroids,labels] = k_means(data, c, 10);
    subplot(1,3,2);
    scatter(data(:,1),data(:,2),50,labels,'filled','LineWidth',2);
    hold on;
    scatter(centroids(:,1),centroids(:,2),50,"r","+",'LineWidth',2);
    title("Standard k-means");

    [sr_centroids, sr_labels] = sr_k_means(data, c, 10);
    subplot(1,3,3);
    scatter(data(:,1),data(:,2),50,sr_labels,'filled','LineWidth',2);
    
    hold on;
    scatter(sr_centroids(:,1),sr_centroids(:,2),50,"r","+",'LineWidth',2);
    title("Spectral-relaxed k-means");

end

% Implement K-means using the alternating procedure
function [centroids,labels] = k_means(data, k, max_iterations)
    % Select k-points for the intial centroids
    centroids = data(randsample(1:length(data),k,false),:);
    for i = 1:max_iterations
        % Form K clusters by assigning points to the closest centroid
        labels = classify(data, centroids);
        % Recompute the centroid of each cluster
        old_centroids = centroids;
        for cx = 1:k
            centroids(cx,:) = mean(data(labels == cx,:),1);
        end
        % If the centroid hasnt changed, the algorithm has converged.
        if (old_centroids == centroids)
            fprintf("converged after %d steps\n",i);
            break
        end
    end
end

function [centroids,labels] = sr_k_means(data, k, max_iterations)
    % Reduce the dimension to k
    [U,~,~] = svd(data);
    % Run standard k-means on the reduced data
    centroids = zeros(k,size(data,2));
    [~, labels] = k_means(U(1:k,:).',k, max_iterations);
    % recover the centroids & label the data accordingly
    for ix = 1:k
        cx = (labels == ix);
        csz = sum(cx);
        centroids(ix,:) = ones(1,csz)*data(cx,:)/csz;
    end
    labels = classify(data, centroids);
end

function labels = classify(data, centroids)
    N = size(data,1);
    c = size(centroids,1);
    distances = zeros(N,c);
    for ix = 1:c
        distances(:,ix) = sum((data-centroids(ix,:)).^2,2);
    end
    [~,labels] = min(distances,[],2);
end

% Implement the spectral relaxation of k-means
% Create a random dataset and compare the k-means and spectral relaxed
% k-means

