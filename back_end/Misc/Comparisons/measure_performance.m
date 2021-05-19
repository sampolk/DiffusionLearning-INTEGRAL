function [performance, ts]= measure_performance(Clusterings, Y)
    % Calculate the NMI between best clustering and GT as well as Total VI Minimizer and GT 
    nmi_temp = zeros(length(Clusterings.K),1);
    for t = 1:length(nmi_temp)
        nmi_temp(t,:) = nmi(Clusterings.Labels(:,t), Y);
    end
    [performance,ts] = max(nmi_temp);

if ~isfield(Clusterings.Hyperparameters, 'K_Known')
    [performance(2), ts(2)] = nmi(Clusterings.Labels(:,Clusterings.TotalVI.Minimizer_Idx  ), Y);
    
end