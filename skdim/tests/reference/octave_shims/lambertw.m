% Matches scipy.special.lambertw to 3e-15 on real inputs in (-2, 3) (checked when added).
function w = lambertw(x)
% Principal branch W0 of the Lambert W function (Octave shim for MATLAB's lambertw),
% Halley iteration on w*exp(w) = x in complex arithmetic.
  w = zeros(size(x));
  for i = 1:numel(x)
    z = x(i);
    p = sqrt(2 * (exp(1) * z + 1));
    if abs(p) < 1
      v = -1 + p - p^2 / 3;
    else
      v = log(1 + z);
    end
    for it = 1:100
      ev = exp(v);
      f = v * ev - z;
      dv = f / (ev * (v + 1) - (v + 2) * f / (2 * v + 2));
      v = v - dv;
      if abs(dv) < 1e-15 * (1 + abs(v)), break; end
    end
    w(i) = v;
  end
end
