%% 
%Makes Object images black
%%
clear all;

imFolders = dir('Object_OG');
imFolders = imFolders(3:end); %removes '.' and '...' from beginning of folder list

for ii = 1:length(imFolders)
    imFiles = dir(['Object_OG\',imFolders(ii).name, '/*.tif']);
    for kk = 1:length(imFiles)
        % Read in original image
        IM = imread(['Object_OG\',imFolders(ii).name, '/', imFiles(kk).name]);
        %convert to matrix
        IM = im2double(IM);
        
        %make object in image a binary mask
        BW = imbinarize(rgb2gray(IM));
        BW = imcomplement(BW);
        BW2 = imfill(BW,'holes');
        BW3 = double(BW2);
        
        %Multiply mask by original image to make background black
        IM2=IM .* BW3;
        
        mkdir('Object\', imFolders(ii).name);
        imwrite(IM2, ['Object\', imFolders(ii).name, '\', imFiles(kk).name(1:end-3), 'jpg']);
    end
end
