function v=StatMap_ST(datM2,lblC,smType)    

    % datM2 is the 2D matrix (class x run)-by-voxel
    % lblC is 1-by-(class x run) cell array of labels
    % smType selects the test for stability (use Pearson)

        switch smType
            case 'anova'
                hFStatMap=@(dM2) anova1(dM2,[],'off');
            case 'Pearson'
                hFStatMap=@(dM2) thisMeanCorr(dM2,smType);
            case 'Spearman'
                hFStatMap=@(dM2) thisMeanCorr(dM2,smType);
%             case 'var'
%                 hFStatMap=@thisVar;
            otherwise
                warning('DistribModelFMRI:Warn',['smType: ' smType ...
                        ' unrecognised, reverting to anova.']);
                hFStatMap=@(dM2) anova1(dM2,[],'off');
        end
    
        nVox=size(datM2,2);

            % Get the number of unique target categories in each
            % training set
        [unqLblC,~,unqTargIx2Orig]=unique(lblC);
        nUnqLbl=length(unqLblC);
        nRepsPerLbl=length(unqTargIx2Orig)/nUnqLbl;
          
        try
            byLblIxM2=zeros(nRepsPerLbl,nUnqLbl);
            for u=1:nUnqLbl
                byLblIxM2(:,u)=find(unqTargIx2Orig==u);
            end
        catch MErr
            error(['TODO:handle this error!, ' ...
                   'probable mismatch in # target categories']);
        end
           
        v=arrayfun(@(a) hFStatMap(reshape(datM2(byLblIxM2,a),...
                                        nRepsPerLbl,nUnqLbl)),...
                                        1:nVox);
        function meanR=thisMeanCorr(dM2,corrType)
            cM2=corr(dM2','type',corrType);
            rTriLg=tril(true(size(dM2)),-1);
            meanR=mean(cM2(rTriLg));
        end
        
%         function v=thisVar(dM2)
%             v=var(dM2(:));
%         end
    end  