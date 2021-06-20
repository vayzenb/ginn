%%
%pads original FaceGen images to make them squares
%%

clear all;

imFolders = dir('FaceGen_OG');
imFolders = imFolders(3:end); %removes '.' and '...' from beginning of folder list

%loops through image folders
for ii = 1:length(imFolders)
    imFiles = dir(['FaceGen_OG\',imFolders(ii).name, '/*.png']);
    for kk = 1:length(imFiles)
        
       IM = imread(['FaceGen_OG\',imFolders(ii).name, '/', imFiles(kk).name]);

       sqrIM = padarray(IM, [0,round((size(IM,1)-size(IM,2))/2)], 0);

       imwrite(sqrIM, ['FaceGen\', imFolders(ii).name, '\', imFiles(kk).name]);
    end
   
end


   