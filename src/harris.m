%******************************************************************************
% DISCLAIMER: All functions used in this program have been referenced and     *
% obtained from mathworks.com examples for the purposes of this assignment.   *
%******************************************************************************

% Gaussian filter parameters.
FILTER_SIZE = 5;
SIGMA = 1;

% Harris corner detection parameters.
K = 0.05;
THRESHOLD = 10000000;

% Load the image and convert it to a grayscale format.
image = open_and_convert('image1.jfif');

% Construct Gaussian filter as defined by the parameters.
Gaussian = fspecial('gaussian', FILTER_SIZE, SIGMA);

% Compute the spatial derivatives using Sobel kernels.
[I_x, I_y] = sobel_spatial_derivation(image, Gaussian);

% Compute the structure tensor setup
[I_x_squared, I_y_squared, I_x_I_y] = structure_tensor_setup ...
(I_x, I_y, Gaussian);

% Compute the corner response for the image.
R = corner_response_calculation(I_x_squared, I_y_squared, I_x_I_y, K);

% Print the corner response for the image.
figure;
imshow(R);

% Compute logical matrix conveying which pixels have a value higher
% than the predefined threshold.
corners = R > THRESHOLD;
figure;
imshow(corners);

% Perform non-maximum suppression on the corner response.
suppressed_R = non_maximum_suppression(R, image);

% Print the suppressed corner response for the image.
figure;
imshow(suppressed_R);

% Compute logical matrix conveying which pixels have a value higher
% than the predefined threshold.
suppressed_corners = suppressed_R > THRESHOLD;

% Compute and superimpose the corners onto the image.
superimpose_corners(suppressed_R, suppressed_corners, image);

%******************************************************************************
% Function for loading and formatting the image to the appropriate graysclae 
% image to enhance computational performance.
function bw_image = open_and_convert(image)
    % Load the image provided into the application.
    rgb_image = imread(image);

    % Convert the image from RGB format to grayscale format for better results.
    bw_image = rgb2gray(rgb_image);
end

% Function for computing the spatial derivatives of an image by using Sobel
% kernels convoluted with a Gaussian filter.
function [I_x, I_y] = sobel_spatial_derivation(image, Gaussian)
    % Construct Sobel kernels for the horizontal and vertical components.
    S_x = fspecial('sobel');
    S_y = S_x';

    % Convolve the Sobel kernel with the Gaussian filter for incorporating
    % a form of edge detection.
    G_x = matrix_convolution(Gaussian, S_x);
    G_y = matrix_convolution(Gaussian, S_y);
    
    % Compute the spatial derivatives by convolving the modified Gaussian 
    % filters with the image in the X and Y directions.
    I_x = matrix_convolution(image, G_x);
    I_y = matrix_convolution(image, G_y);
end

% Function for computing the structure tensor matrix.
function [I_x_squared, I_y_squared, I_x_I_y] = structure_tensor_setup ...
    (I_x, I_y, Gaussian)
    % Compute the individual convolution components for the matrix.
    I_x_squared = matrix_convolution(I_x.^2, Gaussian);
    I_y_squared = matrix_convolution(I_y.^2, Gaussian);
    I_x_I_y = matrix_convolution(I_x .* I_y, Gaussian);
end

% Function for computing the convolution of two matricies X and Y.
% Compared results with conv2 function and produced similar results.
function Z = matrix_convolution(X, Y)
    % Obtain the sizes for both matricies.
    [row_X, column_X] = size(X);
    [row_Y, column_Y] = size(Y);

    % Reflect the second matrix about the y axis by rotation 180 degrees.
    h = rot90(Y, 2);

    % Initialize the resultant matrix to be the same size as X.
    Z = zeros(row_X , column_X);

    % Resize the X matrix to comply with the convolution indicies.
    resized_X = convolution_resize(X, h, row_X, column_X, row_Y, column_Y);

    % Compute the convolution summation as described in lecture material.
    for i = 1 : row_X
        for j = 1 : column_X
            for k = 1 : row_Y
                for l = 1 : column_Y
                    Z(i,j) = Z(i,j) + (resized_X(i-1 + k, j-1 + l) * h(k,l));
                end
            end
        end
    end
end

% Function for resizing a matrix in accordance with the another for convolution.
% Any pixels left outside the convolution zone will be padded with zeroes.
function resized = convolution_resize(X, h, row_X, column_X, row_Y, column_Y)
    % Compute the 2 dimensional window size for the convolution to take place.
    % Compute the 2 dimensional h matrix center in ternms or rows and cols.
    middle = floor((size(h)+1)/2);

    % The top and bottom ends are dictated by the rows of h and Y.
    bottom_end = row_Y - middle(1);
    top_end = middle(1) - 1;

    % The left and right ends are dictated by the columns of h and Y.
    right_end = column_Y - middle(2);
    left_end = middle(2) - 1;

    % Concatenate sizes into one parameter each.
    rows_resized = row_X + top_end + bottom_end;
    columns_resized = column_X + left_end + right_end;

    % Reconstruct the X matrix with the additional four corner ends.
    resized = zeros(rows_resized, columns_resized);

    % Copy over the parameters from the original matrix into the resized one.
    % The addition of 1 is to counteract the possibility of index 0.
    for i = top_end + 1 : top_end + row_X
        for j = left_end + 1 : left_end + column_X
            resized(i,j) = X(i-top_end, j-left_end);
        end
    end
end

% Function for computing the corner response for the image.
function R = corner_response_calculation(I_x_squared, I_y_squared, I_x_I_y, k)
    % Compute the individual convolution components for the matrix.  
    det_M = I_x_squared .* I_y_squared - I_x_I_y.^2;
    trace_M = I_x_squared + I_y_squared;
    
    % Compute the response value as described in the lecture material.
    R = det_M - (k * (trace_M).^2);
end

% Function for performing non-maximum suppression on a corner response image.
function suppressed_R = non_maximum_suppression(R, image)
    % Extracy the number of rows and columns in the image to loop through.
    [rows, columns] = size(image);

    % Set the initial response to be the same as previously computed.
    suppressed_R = R;

    % Loop through the rows and columns for the image.
    % Start at index 2 because index 1 cannot support i-1 and j-1.
    for i = 2:rows - 1
        for j = 2:columns - 1
            % Find the highest pixel in the neighborhood window.
            neighborhood_max = max([R(i+1,j), R(i,j+1), R(i-1,j), R(i,j-1), ... 
            R(i+1,j-1), R(i-1,j+1), R(i+1,j+1), R(i-1,j-1)]);

            % Evaluate whether this max value is higher than the current pixel.
            if (neighborhood_max > R(i,j))
                % If it isn't the highest value, then supress the pixel to 0.
                suppressed_R(i,j) = 0;
            end
        end
    end
end

% Function for superimposing the computed corners onto the image.
function superimpose_corners(suppressed_R, corners, image)
    % Superimpose the suppressed pixels onto the logical coner matrix.
    corner_pixels = suppressed_R .* corners;
    figure;
    imshow(corner_pixels);

    % Locate the positions of the non-zero components of the response. 
    [row_pixels, column_pixels] = find(corners == 1);

    % Superimpose the corner pixels on top of the original image.
    figure;
    imshow(image);
    hold on;
    plot(column_pixels, row_pixels, 'm.');
end
