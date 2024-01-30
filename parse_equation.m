% img = imread("output.png");
img = imread("17formulaire003-equation000.inkml.jpg");
% imtool(img)

imgGrey = rgb2gray(img);
% imtool(imgGrey)

% migth change
mask = imgGrey < 200;

cc = bwlabel(mask,4);
% max(max(cc))

stats = regionprops(cc, "basic");

% show original bounding boxes

% imshow(~mask)
% for i=1:max(max(cc))
%     param = stats(i).BoundingBox;
%     hold on
%     rectangle('Position',[param(1), param(2), param(3), param(4)])
%     hold off
% end

% merge bounding boxes
threshold = 0.5;
charBB = [stats(1).BoundingBox];
idx = 1;
for i=2:max(max(cc))
    lastEndx = charBB(idx,1) + charBB(idx, 3);
    thisEndx = stats(i).BoundingBox(1)+stats(i).BoundingBox(3);
    fprintf("lastEnd: %d, thisStart: %d\n", lastEndx, stats(i).BoundingBox(1))
    if lastEndx > stats(i).BoundingBox(1) % if the end of last range is greater than start of this
        fprintf('yes overlap\n');
        ratio = (min([lastEndx thisEndx])-max([charBB(idx,1) stats(i).BoundingBox(1)]))/min([charBB(idx,3) stats(i).BoundingBox(3)]);
%         ratio = (min([lastEndx thisEndx])-max([charBB(idx,1) stats(i).BoundingBox(1)]))/mean([charBB(idx,3) stats(i).BoundingBox(3)]);
        fprintf('ratio: %d\n', ratio);
        if ratio > threshold % calculate overlaps
            fprintf("yes merge\n")
            charBB(idx, 1) = min([charBB(idx,1) stats(i).BoundingBox(1)]);
            charBB(idx, 2) = min([charBB(idx,2) stats(i).BoundingBox(2)]);
            charBB(idx,3) = max([lastEndx thisEndx]) - min([charBB(idx,1) stats(i).BoundingBox(1)]);
            lastEndy = charBB(idx,2) + charBB(idx, 4);
            thisEndy = stats(i).BoundingBox(2)+stats(i).BoundingBox(4); 
            charBB(idx,4) = max([lastEndy thisEndy]) - min([charBB(idx,2) stats(i).BoundingBox(2)]);
            continue
        end
    end
    % no merge
    charBB = [charBB; stats(i).BoundingBox];
    idx = idx+1;
end

% show final bounding box
imshow(~mask)
temp = size(charBB);
if temp(1) > 1
    for i=1:length(charBB)
        hold on
        rectangle('Position',charBB(i,:))
        hold off
    end
else
    hold on
    rectangle('Position',charBB(1,:))
    hold off
end
% chop images
% for i=1:length(charBB)
%     temp = imcrop(mask,charBB(i,:));
%     imtool(temp)
% end

% mask2 = cc==3;
% imtool(mask2)
