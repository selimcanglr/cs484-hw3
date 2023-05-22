img_idx = linspace(1, 10, 10);

for i=1:length(img_idx)
    path = img_idx(i) + ".jpg";
    img = imread(path);
    img_gray = rgb2gray(img);

    scales = [2 4 8 16];
    orientations = [0 90 180 270];

    gabor_array = gabor(scales, orientations);
    [gabor_mag, gabor_phase] = imgaborfilt(img_gray, gabor_array);

    % Save filtered images
    for j = 1:numel(gabor_array)
        filtered_img = gabor_mag(:, :, j); % Assuming you want to save the magnitude component
        save_path = "gabor_output/" + img_idx(i) + "_filtered_" + j + ".jpg";
        imwrite(filtered_img, save_path);
    end
end