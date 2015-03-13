%BATHYMETRY_FEATURES_MAT_TO_CSV Convert bathymetry feature .mat to CSVs
%    BATHYMETRY_FEATURES_MAT_TO_CSV Loads MATLAB bathymetry feature files
%    and saves the arrays stored in the data structure to CSV files. A CSV
%    file will be created for each feature in the binary file:
%
%        - aspect      -> aspect_<iii>.csv
%        - rugosity    -> rugosity<iii>.csv
%        - slope       -> slope<iii>.csv
%        - processed   -> processed<iii>.csv
%        - neighbours  -> neighbours<iii>.csv
%
%    where <iii> is the scale (number of neighbours) used to calculate the
%    features.
%
clear all; close all; clc;

% Bathymetry
BATHY = '/media/Data/PhD/Sand_Box/matlab/unstable/IROS_2012/data/bathymetry_features/ohara_2008_bathymetry.mat';

% Features to convert.
FILES = {'/media/Data/PhD/Sand_Box/matlab/unstable/IROS_2012/data/bathymetry_features/ohara_2008_bathymetry_002.mat', ...
         '/media/Data/PhD/Sand_Box/matlab/unstable/IROS_2012/data/bathymetry_features/ohara_2008_bathymetry_008.mat', ...
         '/media/Data/PhD/Sand_Box/matlab/unstable/IROS_2012/data/bathymetry_features/ohara_2008_bathymetry_016.mat' };
     
% Output path.
PREFIX = '/media/Data/Code/survey_planning/features/ohara_2008_bathymetry';

% Load bathymetry.
fprintf('Loading: %s\n', BATHY)
load(BATHY)

dlmwrite([PREFIX, '_index.csv'], find(isnan(bathymetry.Z) == false), 'precision', '%i')
dlmwrite([PREFIX, '_depth.csv'], bathymetry.Z(isnan(bathymetry.Z) == false), 'precision', '%1.24f')
dlmwrite([PREFIX, '_x_bins.csv'], bathymetry.X_bins, 'precision', '%1.24f')
dlmwrite([PREFIX, '_y_bins.csv'], bathymetry.Y_bins, 'precision', '%1.24f')
dlmwrite([PREFIX, '_resolution.csv'], bathymetry.resolution, 'precision', '%1.24f')

% Load bathymetry features and convert to CSV files.
for i = 1:length(FILES)
    clear SVDcoeff aspect neighbours processed rugosity slope window_width
    fprintf('\nLoading: %s\n', FILES{i})
    load(FILES{i})
    
    tic();
    fprintf('    writing processed...  ')
    processed = find(processed) - 1;
    dlmwrite([PREFIX, sprintf('_processed_%03i.csv', neighbours)], ...
             processed, 'precision', '%i')
    toc();
    
    tic();
    fprintf('    writing aspect...     ')
    aspect = aspect(isnan(aspect) == false);
    dlmwrite([PREFIX, sprintf('_aspect_%03i.csv', neighbours)], ...
             aspect, 'precision', '%1.24f')
    toc();
         
    tic();
    fprintf('    writing rugosity...   ')
    rugosity = rugosity(isnan(rugosity) == false);
    dlmwrite([PREFIX, sprintf('_rugosity_%03i.csv', neighbours)], ...
             rugosity, 'precision', '%1.24f')
    toc();
         
    tic();
    fprintf('    writing slope...      ')
    slope = slope(isnan(slope) == false);
    dlmwrite([PREFIX, sprintf('_slope_%03i.csv', neighbours)], ...
             slope, 'precision', '%1.24f')
    toc();
end