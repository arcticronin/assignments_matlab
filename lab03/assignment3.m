%% read images
clear all;

% template
boxImage = imread("elephant.jpg");

% desk
sceneImage = imread("clutteredDesk.jpg");

%%  keypoint detection
tic

boxPoints = detectSURFFeatures(boxImage, "NumOctaves", 8, "MetricThreshold", 200);
scenePoints = detectSURFFeatures(sceneImage, "NumOctaves", 8, "MetricThreshold", 200);
figure(1), clf
imshow (boxImage), hold on
plot(selectStrongest(boxPoints, 100)), hold off

figure(2), clf
imshow (sceneImage), hold on
plot(selectStrongest(scenePoints, 100)), hold off

%% keypoint decription
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

% Match features
boxPairs = matchFeatures(boxFeatures, sceneFeatures, 'MatchThreshold', 50, 'MaxRatio', 0.7);
%boxPairs = matchFeatures(boxFeatures, sceneFeatures);
matchedBoxPoints = boxPoints(boxPairs(:,1),:);
matchedScenePoints = scenePoints(boxPairs(:,2),:);
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');

%% geometric consistency check

[tform, inlierBoxPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedBoxPoints, ...
    matchedScenePoints, 'affine');
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');

%% bounding box drawing (draw the box)

boxPoly = [1 1;
    size(boxImage, 2) 1;
    size(boxImage, 2) size(boxImage, 1);
    1 size(boxImage, 1);
    1 1];

newBoxPoly = transformPointsForward(tform, boxPoly);
figure, clf
imshow(sceneImage), hold on
line(newBoxPoly(:, 1), newBoxPoly(:,2), 'Color', 'y');
hold off
toc

%% more precise bounding box
figure, clf
imshow(boxImage), hold on
[x,y]=ginput(12);
x=[x; x(1)];
y=[y; y(1)];
newBoxPoly=transformPointsForward(tform,[x y]);
hold off

figure, clf
imshow(sceneImage), hold on
line(newBoxPoly(:,1),newBoxPoly(:,2),'Color','y')
hold off
toc

%%
figure, clf
imshow(boxImage), hold on
line(x,y, 'color','r')
hold off




