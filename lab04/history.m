%% Parameters used with HOG
% final ap = 0.4606
%trainCascadeObjectDetector('myFaceDetector.xml', positiveInstances, negativeImages,'FalseAlarmRate',0.07,'NumCascadeStages',5, 'NegativeSamplesFactor',2, 'TruePositiveRate',0.999);%, 'FeatureType','LBP');
% 
% Automatically setting ObjectTrainingSize to [32, 32]
% Using at most 6686 of 6713 positive samples per stage
% Using at most 13372 negative samples per stage
% 
% --cascadeParams--
% Training stage 1 of 5
% [........................................................................]
% Used 6686 positive and 13372 negative samples
% Time to train stage 1: 6 seconds
% 
% Training stage 2 of 5
% [........................................................................]
% Used 6686 positive and 13372 negative samples
% Time to train stage 2: 25 seconds
% 
% Training stage 3 of 5
% [........................................................................]
% Used 6686 positive and 13372 negative samples
% Time to train stage 3: 48 seconds
% 
% Training stage 4 of 5
% [........................................................................]
% Used 6686 positive and 13372 negative samples
% Time to train stage 4: 162 seconds
% 
% Training stage 5 of 5
% [........................................................................]
% Used 6686 positive and 13372 negative samples
% Time to train stage 5: 376 seconds
% 
% Training complete


%% Same but with LBP scores ap=0.6227
% Training stage 1 of 5
% [........................................................................]
% Used 6581 positive and 13162 negative samples
% Time to train stage 1: 31 seconds
% 
% Training stage 2 of 5
% [........................................................................]
% Used 6581 positive and 13162 negative samples
% Time to train stage 2: 55 seconds
% 
% Training stage 3 of 5
% [........................................................................]
% Used 6581 positive and 13162 negative samples
% Time to train stage 3: 70 seconds
% 
% Training stage 4 of 5
% [........................................................................]
% Used 6581 positive and 7943 negative samples
% Time to train stage 4: 91 seconds
% 
% Training stage 5 of 5
% [....................................................Warning:
% Unable to generate a sufficient number of negative samples for this stage.
% Consider reducing the number of stages, increasing the false alarm rate
% or adding more negative images. 
% 
% Cannot find enough samples for training. 
% Training will halt and return cascade detector with 4 stages
% Time to train stage 5: 46 seconds
% 
% Training complete


%using Haar I can't finish training --> too complex


%% Augmented tests
% 
% trainCascadeObjectDetector('myFaceDetector.xml', positiveInstances, negativeImages,'FalseAlarmRate',0.4,'NumCascadeStages',15, 'FeatureType','LBP');
% 
% 
% Training stage 1 of 15
% [........................................................................]
% Used 25095 positive and 50190 negative samples
% Time to train stage 1: 125 seconds
% 
% Training stage 2 of 15
% [........................................................................]
% Used 25095 positive and 50190 negative samples
% Time to train stage 2: 146 seconds
% 
% Training stage 3 of 15
% [........................................................................]
% Used 25095 positive and 50190 negative samples
% Time to train stage 3: 184 seconds
% 
% Training stage 4 of 15
% [........................................................................]
% Used 25095 positive and 50190 negative samples
% Time to train stage 4: 229 seconds
% 
% Training stage 5 of 15
% [........................................................................]
% Used 25095 positive and 50190 negative samples
% Time to train stage 5: 251 seconds
% 
% Training stage 6 of 15
% [........................................................................]
% Used 25095 positive and 50190 negative samples
% Time to train stage 6: 300 seconds
% 
% Training stage 7 of 15
% [........................................................................]
% Used 25095 positive and 50190 negative samples
% Time to train stage 7: 388 seconds
% 
% Training stage 8 of 15
% [........................................................................]
% Used 25095 positive and 35394 negative samples
% Time to train stage 8: 429 seconds
% 
% Training stage 9 of 15
% [........................................................................]
% Used 25095 positive and 14443 negative samples
% Time to train stage 9: 336 seconds
% 
% Training stage 10 of 15
% [........................................................................]
% Used 25095 positive and 6970 negative samples
% Time to train stage 10: 290 seconds
% 
% Training stage 11 of 15
% [....................................................Warning:
% Unable to generate a sufficient number of negative samples for this stage.
% Consider reducing the number of stages, increasing the false alarm rate
% or adding more negative images. 
% > In trainCascadeObjectDetector (line 314)
% In assignment_04 (line 55) 
% 
% Cannot find enough samples for training. 
% Training will halt and return cascade detector with 10 stages
% Time to train stage 11: 161 seconds

%% WITHOUT AUGMENTATIONS
% trainCascadeObjectDetector('myFaceDetector.xml', positiveInstances, negativeImages,'FalseAlarmRate',0.3,'NumCascadeStages',10, 'FeatureType','LBP');
% last-myfacedetector