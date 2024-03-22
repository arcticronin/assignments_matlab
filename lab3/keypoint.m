%% read images

% template
boxImage = imread("stapleRemover.jpg");

% desk
sceneImage = imread("clutteredDesk.jpg");

% clf clear contentof figure
% figure(1), clf, imshow(boxImage);
% figure(2), clf, imshow(sceneImage);

figure(1), clf, imagesc(boxImage);
figure(2), clf, imagesc(sceneImage);

%% we can also crop download it

% boxImage2 = imcrop(boxImage);

%% compute scale factor
% so that we can perform sliding window with a fixed scale we compute it 
% (manually) as te ratio og the same box dimension in the two images

% it's more or less 3 times smaller, in the template that in the image
fs = 2.82;
boxImage = imresize(boxImage, 1/fs);
% boxImage = imresize(boxImage, [1,2]);

% figure(1), clf, imagesc(boxImage);



%%  keypoint detection
tic

boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);

figure(1), clf
imshow (boxImage), hold on
plot(selectStrongest(boxPoints, 100)), hold off

figure(2), clf
imshow (sceneImage), hold on
plot(selectStrongest(scenePoints, 100)), hold off

%% keypoint decription
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

boxPairs = matchFeatures(boxFeatures, sceneFeatures);
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
imshow(boxImage);

% get coordinate from the mouse!! noaise
[x, y] = ginput(4);

%%

x = [x ; x(1)];
y = [y ; y(1)];

newBoxPoly = transformPointsForward(tform, [x,y]);

figure, clf

imshow(sceneImage), hold on

line(newBoxPoly(:, 1), newBoxPoly(:,2), 'color', 'r');
hold off


%% assignment
% 1) run as is with elephant
% 2) change parameters in detectSURFFeaturesmaybe returning more points can be good
% 3) also matchfeatures can be used to get more points
% 4) also estimate geometricTransform can be parametrize
% 5) also changing the shape of the box around the elephant: more than 5
% points polygons would work