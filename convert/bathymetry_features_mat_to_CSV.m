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
BATHY = '/media/Data/PhD/Sand_Box/data/bathymetry/RAW_DATA/GA/Scott_Reef/scott_reef_ascii_grid_update_20090806.mat';

% Features to convert.
FILES = {'/media/Data/PhD/Sand_Box/data/bathymetry/RAW_DATA/GA/Scott_Reef/features/scott_reef_ascii_grid_update_20090806_001.mat', ...
         '/media/Data/PhD/Sand_Box/data/bathymetry/RAW_DATA/GA/Scott_Reef/features/scott_reef_ascii_grid_update_20090806_002.mat', ...
         '/media/Data/PhD/Sand_Box/data/bathymetry/RAW_DATA/GA/Scott_Reef/features/scott_reef_ascii_grid_update_20090806_004.mat', ...
         '/media/Data/PhD/Sand_Box/data/bathymetry/RAW_DATA/GA/Scott_Reef/features/scott_reef_ascii_grid_update_20090806_008.mat', ...
         '/media/Data/PhD/Sand_Box/data/bathymetry/RAW_DATA/GA/Scott_Reef/features/scott_reef_ascii_grid_update_20090806_016.mat' };
     
% Output path.
PREFIX = '/media/Data/Code/survey_planning/data/bathymetry/scott_reef';

% Load bathymetry.
fprintf('Loading: %s\n', BATHY)
load(BATHY)


% Define transverse mercator map projection.
mstruct = defaultm('tranmerc');
mstruct.falsenorthing = 10000000;
mstruct.falseeasting  = 500000;
mstruct.geoid         = [6378137 0.0818];
mstruct.origin        = [0 123 0];
mstruct.scalefactor   = 0.9996;
mstruct = defaultm( mstruct );

% Get X/Y-bins in lat/long.
[x_locs, y_locs] = meshgrid(bathymetry.X_bins, bathymetry.Y_bins);
[lat_grid, lon_grid] = minvtran(mstruct, x_locs, y_locs);
lat_bins = lat_grid(:, 1);
lon_bins = lon_grid(1, :);
zone = utmzone(lat_bins(1), lon_bins(1));

dlmwrite([PREFIX, '_zone.csv'], zone)
dlmwrite([PREFIX, '_lat_bins.csv'], lat_bins, 'precision', '%1.24f')
dlmwrite([PREFIX, '_lon_bins.csv'], lon_bins, 'precision', '%1.24f')


dlmwrite([PREFIX, '_index.csv'], find(isnan(bathymetry.Z) == false) - 1, 'precision', '%i')
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
    dlmwrite([PREFIX, sprintf('_index_%03i.csv', neighbours)], ...
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
