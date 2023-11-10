function Rm_Sli_Reli(InputFile,DelImg,TR,SInd,RInd,NumPasses)
[RmFile]=gretna_RUN_RmFstImg(InputFile, DelImg);
gretna_RUN_SliTim(RmFile, TR, SInd, RInd);
[a,b,~]=fileparts(RmFile{1});
SliFile=[a filesep 'a' b '.nii'];
SliFile={SliFile};
gretna_RUN_Realign(SliFile, NumPasses);
