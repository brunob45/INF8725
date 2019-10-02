%% Nettoyage du Workspace
    clear all;
    close all;
    clc;

%% Exercice 1

    %%Numero 1
    
        img = imread('images\theArtist.png');        
        
        figure;
        imshow(img);
        
        img = Egalisation_Histogramme(img);
        figure;
        imshow(uint8(img));
        
    %%Numero 2
        % voir fonction Convolution
        
    %%Numero 3
        
        gaussien = (1/90) * [1,2,1,2,1;
                             2,4,8,4,2;
                             1,8,18,8,1;
                             2,4,8,4,2;
                             1,2,1,2,1];
        
        img = Convolution(img, gaussien);
        
        figure;
        imshow(uint8(img));
        
    %%Numero 4
        
        [img, lap] = Rehaussement_Contour(img,0.3);
        
        for i=1:size(img,1)
            for j=1:size(img,2)
                lap(i,j) = max(lap(i,j), 0);
                img(i,j) = max(img(i,j), 0);
            end
        end
        
        figure;
        subplot(1,2,1);
        imshow(uint8(lap));
        subplot(1,2,2);
        imshow(uint8(img));
        
    %%Numero 5
        % Lorsqu'on rehausse les contours de l'image, nous faisons
        % r�apparaitre du bruit sel dans l'image. Pour contrer ce bruit,
        % nous pourrions utiliser un filtre "Median".
        
%% Exercice 2
    clear all;
    close all;
    clc;
    
    %%Numero 1
        img = rgb2gray(imread('images\pieces.jpg'));
        imshow(uint8(img));
        img = Binariser(img, 250);
        figure;
        imshow(uint8(img));
        
    %%Numero 2
        img = imclose(img, strel('disk',10));
        figure;
        imshow(uint8(img));
        
    %%Numero 3
        monnaie = Compter_Monnaie(img);
        % la valeur de retour est un array contenant le nombre pour chaque
        % piece de monnaie [5, 10, 25, 2.00]
        
%% Exercice 3
    %%Numero 1
        
%% Fontions

function[result] = Egalisation_Histogramme(img)
        [h,count] = imhist(img);
        hnorm = h/(size(img,1)*size(img,2));
        
        result = zeros(548,1000);
        
        for i=1:size(img,1)
            for j=1:size(img,2)
                sum = 0;
                for k=1:img(i,j)
                    sum = sum + hnorm(k);
                end
                result(i,j) = 255*sum;
            end
        end
end

function[result] = Convolution(img, masque)
    m = uint64(size(masque,1)/2);
    n = uint64(size(masque,2)/2);
    
    masque = rot90(masque, 2);
    result = zeros(size(img,1),size(img,2));
    
    for i=m:(size(img,1)-m+1)
        for j=n:(size(img,2)-n+1)
            submatrix = img(i-m+1:i+m-1, j-n+1:j+n-1);
            mat = double(submatrix).*masque;
            s = sum(mat, 'all');
            result(i,j) = s;
        end
    end
end

function[result, lap] = Rehaussement_Contour(img,k)
        
        gaussien = (1/16) * [1,2,1;
                             2,4,2;
                             1,2,1];
                         
        laplacien = [-1,-1,-1;
                     -1,8,-1;
                     -1,-1,-1];
                 
        lap = Convolution(img,laplacien);
        
        result = Convolution(img,gaussien) + k * lap;
end

function[result] = Binariser(img, k)
    result = zeros(size(img,1),size(img,2));
    for i=1:size(img,1)
        for j=1:size(img,2)
            if img(i,j) > k
                result(i,j) = 0;
            else
                result(i,j) = 255;
            end
        end
    end
end

function[result] = Compter_Monnaie(img)
    img = imerode(img, strel('disk',80)); %80
    [L,dix] = bwlabel(img);

    img = imerode(img, strel('disk',20)); %100
    [L,cinq] = bwlabel(img);

    img = imerode(img, strel('disk',15)); %115
    [L,vcinq] = bwlabel(img);

    img = imerode(img, strel('disk',25)); %130
    [L,deux] = bwlabel(img);

    img = imerode(img, strel('disk',20)); %150
    [L,cookie] = bwlabel(img);

    dix = dix - cinq;
    cinq = cinq - vcinq;
    vcinq = vcinq - deux;
    deux = deux - cookie;
    result = [cinq, dix, vcinq, deux];
end