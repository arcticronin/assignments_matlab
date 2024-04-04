%% read images

clear all
close all
clc

% template
boxImage = imread("elephant.jpg");

% desk
sceneImage = imread("clutteredDesk.jpg");

J = imhistmatch(boxImage,sceneImage);
%J = histeq(sceneImage);


% clf clear contentof figure
%figure(1), clf, imshow(boxImage);
%figure(2), clf, imshow(sceneImage);

figure(1), clf, imagesc(boxImage);
figure(2), clf, imagesc(sceneImage);
figure(3), clf, imagesc(J);
hold off;
boxImage=J;
figure(4);

%% we can also crop download it

boxImage = imcrop(boxImage);

%% compute scale factor
% so that we can perform sliding window with a fixed scale we compute it 
% (manually) as te ratio og the same box dimension in the two images

% it's more or less 3 times smaller, in the template that in the image
% fs = 2.82;
% boxImage = imresize(boxImage, 1/fs);
% boxImage = imresize(boxImage, [1,2]);

% figure(1), clf, imagesc(boxImage);

%% sliding window init

% mind also to not make the coordinate go negative
boxImage = im2double(boxImage);
sceneImage = im2double(sceneImage);

Sb = size(boxImage);
Ss = size(sceneImage);

step = 2; % 1 is every pixel, 5 is skipping 5 every time
Map = [];

%% sliding: 2 nested cycles, 1 for the rows 1 for the columns

tic
for rr = 1 : step : (Ss(1)-Sb(1))
    tmp = [];
    for cc = 1 : step : (Ss(2)-Sb(2))
        D = sceneImage(rr : ...
            rr + Sb(1) - 1 , ...
            cc : ...
            cc + Sb(2) - 1) - boxImage;
        D = D.^2;
        D = sum(D, 'all'); % <-- Sum of Squared Differences
        tmp = [tmp D];
    end
    Map = [Map; tmp];
    figure(3), clf, imagesc(Map), colorbar, drawnow; % just to see the result
end
toc

%figure(5), clf, imagesc(sceneImage);
%figure(4), clf, imagesc(boxImage)