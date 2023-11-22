function [MeanSNR]=SignalToNoiseRatio(InputFile,MaskFile)
    [Data,~,~]=y_ReadRPI(InputFile);
    [Rows,Columns,Slices,Timepoints]=size(Data);
    DataDemension=[Rows,Columns,Slices];
    Data=reshape(Data,[],Timepoints);
    SNR=mean(Data,2)./(std(Data'))';
    % SNR=(std(Data'))'./mean(Data,2);

    [MaskData,~,~]=y_ReadRPI(MaskFile);
    [RowsM,ColumnsM,SlicesM]=size(MaskData);
    MDataDemension=[RowsM,ColumnsM,SlicesM];
    MaskData=reshape(MaskData,[],1);
    if ~isequal(DataDemension,MDataDemension)
        error('The InputFile Data Demension is not consistent with MaskData');
    end

    MaskedSNR=SNR(find(MaskData==1));
    MeanSNR=meanabs(MaskedSNR);
end
