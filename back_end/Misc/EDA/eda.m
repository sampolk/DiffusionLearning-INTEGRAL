function eda(x, log_on)

if nargin == 1
    log_on = 1;
end

if log_on
    try
        imagesc(log10(reshape(x,500,500)))
    catch
        imagesc((reshape(x,500,500)))
    end
else
    imagesc((reshape(x,500,500)))
end
colorbar
xticks([])
yticks([])
pbaspect([1,1,1])