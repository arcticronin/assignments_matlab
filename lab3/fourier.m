%% read images

% template
boxImage = imread("elephant.jpg");

% desk
sceneImage = imread("clutteredDesk.jpg");

% clf clear contentof figure
%figure(1), clf, imshow(boxImage);
%figure(2), clf, imshow(sceneImage);

figure(1), clf, imagesc(boxImage);
figure(2), clf, imagesc(sceneImage);

%% we can also crop

boxImage = imcrop(boxImage);


% Pad the template to match the scene size
paddedBoxImage = zeros(size(sceneImage));
paddedBoxImage(1:size(boxImage,1), 1:size(boxImage,2)) = boxImage;

% Apply FFT to both images
fftScene = fft2(sceneImage);
fftBox = fft2(paddedBoxImage, size(sceneImage,1), size(sceneImage,2)); % Ensure same size

% Perform multiplication in the frequency domain
product = fftScene .* conj(fftBox);

% Apply inverse FFT
correlationMap = ifft2(product);

% Find the maximum correlation to locate the template, at coord x y
[maxCorr, ind] = max(abs(correlationMap(:)));
[y, x] = ind2sub(size(correlationMap), ind);

% Display the scene image
figure, imshow(sceneImage, []);
hold on; 

% Calculate the dimensions of the template
[templateHeight, templateWidth] = size(boxImage);

% rectangle('Position', [x, y, width, height], 'EdgeColor', 'r', 'LineWidth', 2)
rectangle('Position', [x, y, templateWidth, templateHeight], 'EdgeColor', 'r', 'LineWidth', 2);

hold off;