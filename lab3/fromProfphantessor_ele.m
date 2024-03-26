clear all
clc

%% read images
% template
boxImage = imread('./elephant.jpg');

% desk
sceneImage = imread('./clutteredDesk.jpg');

figure(1), clf, imshow(boxImage)
figure(2), clf, imshow(sceneImage)
% figure(1), clf, imagesc(boxImage)
% figure(2), clf, imagesc(sceneImage)

tic
%% keypoint detection
boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);

figure(1), clf
imshow(boxImage), hold on
plot(selectStrongest(boxPoints,100)), hold off

figure(2), clf
imshow(sceneImage), hold on
plot(selectStrongest(scenePoints,100)), hold off

%% keypoint description
[boxFeatures, boxPoints]=extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints]=extractFeatures(sceneImage, scenePoints);

%% feature matching
boxPairs = matchFeatures(boxFeatures, sceneFeatures);
matchedBoxPoints = boxPoints(boxPairs(:,1),:);
matchedScenePoints = scenePoints(boxPairs(:,2),:);
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');

%% geometric consistency check
[tform, inlierBoxPoints, inlierScenePoints]=...
    estimateGeometricTransform(matchedBoxPoints,...
    matchedScenePoints,'affine');
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');

%% bounding box drawing
boxPoly = [1 1;
    size(boxImage,2) 1;
    size(boxImage,2) size(boxImage,1);
    1 size(boxImage,1);
    1 1];
newBoxPoly=transformPointsForward(tform,boxPoly);
figure, clf
imshow(sceneImage), hold on
line(newBoxPoly(:,1),newBoxPoly(:,2),'Color','y')
hold off
toc

%% more precise bounding box
figure, clf
imshow(boxImage)
[x,y]=ginput(4);
%%
x=[x; x(1)];
y=[y; y(1)];
newBoxPoly=transformPointsForward(tform,[x y]);
figure, clf
imshow(sceneImage), hold on
line(newBoxPoly(:,1),newBoxPoly(:,2),'Color','y')
hold off
toc