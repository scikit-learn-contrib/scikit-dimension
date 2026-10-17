% Reference values from the original MATLAB code, run in Octave by generate_fixtures.py:
%   octave-cli reference.m <indir>
% <indir> holds the downloaded .m files and, per dataset <name>: <name>_X.csv, <name>_idx.csv
% (1-based kNN indices, self excluded) and <name>_dists.csv. Writes <name>_out_*.csv.
% conda-forge's octave-statistics installs files without registering the package: add them to the path
stat = dir(fullfile(OCTAVE_HOME(), 'share', 'octave', 'packages', 'statistics-*'));
addpath(fullfile(stat(end).folder, stat(end).name));
indir = argv(){1};
addpath(fullfile(fileparts(mfilename('fullpath')), 'octave_shims'));  % lambertw, missing in Octave
addpath(indir);
files = dir(fullfile(indir, '*_X.csv'));
for f = 1:numel(files)
  name = files(f).name(1:end-6);
  X = csvread(fullfile(indir, [name '_X.csv']));
  idx = csvread(fullfile(indir, [name '_idx.csv']));
  dists = csvread(fullfile(indir, [name '_dists.csv']));
  n = size(X, 1);
  tle = zeros(n, 1);
  for i = 1:n
    % skdim's TLE is the variant with central measurements and reflections (idtlec), default epsilon 1e-4
    tle(i) = idtlec(X(idx(i, :), :), dists(i, :), 1e-4);
  end
  mom = idmom(dists')';
  [n_alpha, n_single] = SeparabilityAnalysis(X, 'ProducePlots', 0);
  csvwrite(fullfile(indir, [name '_out_tle.csv']), tle);
  csvwrite(fullfile(indir, [name '_out_mom.csv']), mom);
  dlmwrite(fullfile(indir, [name '_out_fisher.csv']), [n_single, n_alpha(:)'], 'precision', 17);
end
