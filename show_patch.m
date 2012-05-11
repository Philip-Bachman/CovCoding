function [ patch ] = show_patch( patch, rows, cols )
%SHOW_PATCH Summary of this function goes here
%   Detailed explanation goes here

patch = double(patch);
patch = patch - min(patch);
patch = patch / max(patch);

if (numel(patch) > rows*cols)
    patch = reshape(patch,rows,cols,3);
else
    patch = reshape(patch,rows,cols);
end

imshow(patch, 'DisplayRange', [], 'InitialMagnification', 300);

end

