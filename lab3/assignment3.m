%% read images
clear all;

% template
boxImage = imread("elephant.jpg");

% desk
sceneImage = imread("clutteredDesk.jpg");

%%  keypoint detection
tic

boxPoints = detectSURFFeatures(boxImage, "NumOctaves", 8, "MetricThreshold", 50);
scenePoints = detectSURFFeatures(sceneImage, "NumOctaves", 8, "MetricThreshold", 50);

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

%% assignment
% 1)[x] run as is with elephant
% 2)[x] change parameters in detectSURFFeaturesmaybe returning more points can be good
% 3)[x] added modality ORB
% 3)[x] also matchfeatures can be used to get more points
% 4)[] also estimate geometricTransform can be parametrize
% 5)[x] also changing the shape of the box around the elephant: more than 5
% points
% keypoints from stape remover = 389
% keypoints same code with eleph = 272
% 
% after parametrization 898 keypoints
% boxPoints = detectSURFFeatures(boxImage, "NumOctaves", 8, "MetricThreshold", 50);


