% Apply PCA to USPS.mat
function recerr = hw5_2
    usps = load('USPS.mat');
    A = usps.A;
    figure;
    plist = [10 50 100 200];
    %imlist = randsample(1:length(A),4);
    imlist = [1 2];
    recerr = zeros(length(plist),1);
    for px = 1:length(plist)
        p = plist(px);
        [U, Sigma, V] = svd(A - mean(A),0);
        reconstructed = U * Sigma(:,1:p) * V(:,1:p)';
        recerr(px) = norm(A-reconstructed, 'fro')^2;
        for ix=1:length(imlist)
            subplot(length(imlist),length(plist), (ix-1)*length(plist)+px);
            img = reshape(reconstructed(imlist(ix),:),16,16)';
            imagesc(img);
            set(gca,'xtick',[]);
            set(gca,'ytick',[]);
        end
    end
    colormap("gray");
    
end