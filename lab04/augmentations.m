
%% augmenting negative faces

outdir = 'CaltechFaces/my2_train_non_face_scenes/';
neg=dir('./CaltechFaces/my_train_non_face_scenes/*.jpg');

for ii = 1:size(neg,1)
    im = imread([neg(ii).folder filesep neg(ii,1).name]);
    imwrite(im, [outdir filesep neg(ii,1).name]);

    [pp,ff,ee] = fileparts(neg(ii).name);
    
    % Horizontal flip
    im_flip = fliplr(im);
    imwrite(im_flip, [outdir filesep ff '_flip' ee]);
    
    % Rotation
    for nrot = 1:50
        imr = imrotate(im, 180*rand(1)-90, 'crop');
        imwrite(imr, [outdir filesep ff '_r' num2str(nrot) ee]);
   
        % Scaling
        scaleFactor = 0.9 + (1.1-0.9).*rand(); % Random scale between 90% to 110%
        im_scaled = imresize(im, scaleFactor);
        imwrite(im_scaled, [outdir filesep ff '_scaled' num2str(nrot) ee]);
        
        % Noise injection
        im_noise = imnoise(im, 'gaussian', 0, 0.05 .* rand()); % Adds Gaussian noise
        imwrite(im_noise, [outdir filesep ff '_noise' num2str(nrot) ee]);
    
    end
    % Cropping
    [rows, cols, ~] = size(im);
    cropSize = round([rows * 0.8, cols * 0.8]); % Crop size 80% of original
    startRow = randi([1, rows - cropSize(1) + 1]);
    startCol = randi([1, cols - cropSize(2) + 1]);
    im_cropped = imcrop(im, [startCol, startRow, cropSize(2)-1, cropSize(1)-1]);
    imwrite(im_cropped, [outdir filesep ff '_cropped' ee]);

    % Brightness adjustment
    im_bright = imadjust(im, stretchlim(im), []); % Auto adjusts the intensity values
    imwrite(im_bright, [outdir filesep ff '_bright' ee]);

    
end

%% Augrentation of positive faces
mkdir('CaltechFaces/my2_train_faces/');
outdir = 'CaltechFaces/my2_train_faces/';
faces = dir('./CaltechFaces/my_train_faces/*.jpg');
for ii = 1:size(faces,1)
    im = imread([faces(ii).folder filesep faces(ii).name]);
    imwrite(im, [outdir filesep faces(ii,1).name]);

    [pp,ff,ee] = fileparts(faces(ii).name);
    
    % Horizontal flip - good for faces as it simulates mirror images
    im_flip = fliplr(im);
    imwrite(im_flip, [outdir filesep ff '_flip' ee]);
    
    % % Rotation - small rotations as drastic ones might not be practical for faces
    % for nrot = 1:5  % Less rotations for faces, considering practical utility
    %     angle = -15 + (15+15)*rand(); % Random angle between -15 to 15 degrees
    %     imr = imrotate(im, angle, 'bilinear', 'crop');
    %     imwrite(imr, [outdir filesep ff '_r' num2str(nrot) ee]);
    % end

    % % Scaling - slight scaling to simulate faces being closer or further away
    % scaleFactor = 0.95 + (1.05-0.95).*rand(); % Random scale between 95% to 105%
    % im_scaled = imresize(im, scaleFactor);
    % imwrite(im_scaled, [outdir filesep ff '_scaled' ee]);

    % Brightness adjustment - very useful for faces to handle different lighting
    im_bright = imadjust(im, stretchlim(im), []);
    imwrite(im_bright, [outdir filesep ff '_bright' ee]);

    % Noise injection - to simulate grainy images or low-quality cameras
    im_noise = imnoise(im, 'gaussian', 0, 0.005); % Lower noise level than in non-faces
    imwrite(im_noise, [outdir filesep ff '_noise' ee]);
end

