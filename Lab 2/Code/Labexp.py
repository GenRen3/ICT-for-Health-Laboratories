#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy


count = 0
np.set_printoptions(precision=2)  # use only two decimal digits when printing numbers
plt.close('all')  # close previously opened pictures
# 4 color levels filein='low_risk_10.jpg';# file to be analyzed
# 4 color levels filein='melanoma_21.jpg';# file to be analyzed
# 6 color levels filein='melanoma_27.jpg';# file to be analyzed
# 4 color levels filein='medium_risk_1.jpg';# file to be analyzed
for number in range(27, 28):
    image = 'melanoma_'+str(number)
    file = './moles/'+image+'.jpg'  # file to be analyzed
    image2 = 'movie_'+str(count)
    filecleaning = './moles/pdf/'+image2+'.pdf'
    # file = './moles/melanoma_22.jpg'
    im_or = mpimg.imread(file)
    # im_or is Ndarray 583 x 584 x 3 unint8
    # plot the image, to check it is correct:
    plt.figure()
    plt.imshow(im_or)
    plt.title('original image')
    # plt.draw()
    plt.pause(0.1)
    # %% reshape the image from 3D to 2D
    N1, N2, N3 = im_or.shape  # note: N3 is 3, the number of elementary colors, i.e. red, green ,blue
    # im_or(i,j,1) stores the amount of red for the pixel in position i,j
    # im_or(i,j,2) stores the amount of green for the pixel in position i,j
    # im_or(i,j,3) stores the amount of blue for the pixel in position i,j
    # we resize the original image
    im_2D = im_or.reshape((N1*N2, N3))  # im_2D has N1*N2 row and N3 columns
    # pixel in position i.j goes to position k=(i-1)*N2+j)
    # im_2D(k,1) stores the amount of red of pixel k
    # im_2D(k,2) stores the amount of green of pixel k
    # im_2D(k,3) stores the amount of blue of pixel k
    # im_2D is a sequence of colors, that can take 2^24 different values
    Nr, Nc = im_2D.shape
    # %% get a simplified image with only Ncluster colors
    # number of clusters/quantized colors we want to have in the simpified image:
    Ncluster = 6
    # instantiate the object K-means:
    kmeans = KMeans(n_clusters=Ncluster, random_state=0)
    # run K-means:
    kmeans.fit(im_2D)
    # get the centroids (i.e. the 3 colors). Note that the centroids
    # take real values, we must convert these values to uint8
    # to properly see the quantized image
    kmeans_centroids = kmeans.cluster_centers_.astype('uint8')
    # copy im_2D into im_2D_quant
    im_2D_quant = im_2D.copy()
    for kc in range(Ncluster):
        quant_color_kc = kmeans_centroids[kc, :]
        # kmeans.labels_ stores the cluster index for each of the Nr pixels
        # find the indexes of the pixels that belong to cluster kc
        ind = (kmeans.labels_ == kc)
        # set the quantized color to these pixels
        im_2D_quant[ind, :] = quant_color_kc
    im_quant = im_2D_quant.reshape((N1, N2, N3))
    plt.figure()
    plt.imshow(im_quant, interpolation=None)
    plt.title('image with quantized colors')
    # plt.draw()
    plt.pause(0.1)
    # %% Find the centroid of the main mole

    # %% Preliminary steps to find the contour after the clustering
    #
    # 1: find the darkest color found by k-means, since the darkest color
    # corresponds to the mole:
    centroids = kmeans_centroids
    sc = np.sum(centroids, axis=1)
    i_col = sc.argmin()  # index of the cluster that corresponds to the darkest color
    # 2: define the 2D-array where in position i,j you have the number of
    # the cluster pixel i,j belongs to
    im_clust = kmeans.labels_.reshape(N1, N2)
    # plt.matshow(im_clust)
    # 3: find the positions i,j where im_clust is equal to i_col
    # the 2D Ndarray zpos stores the coordinates i,j only of the pixels
    # in cluster i_col
    zpos = np.argwhere(im_clust == i_col)
    # 4: ask the user to write the number of objects belonging to
    # cluster i_col in the image with quantized colors

    N_spots_str = input("How many distinct dark spots can you see in the image? ")
    plt.close('all')
    N_spots = int(N_spots_str)

    # 5: find the center of the mole
    if N_spots == 1:
        center_mole = np.median(zpos, axis=0).astype(int)
    else:
        # use K-means to get the N_spots clusters of zpos
        kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
        kmeans2.fit(zpos)
        centers = kmeans2.cluster_centers_.astype(int)
        # the mole is in the middle of the picture:
        center_image = np.array([N1//2, N2//2])
        center_image.shape = (1, 2)
        d = np.zeros((N_spots, 1))
        for k in range(N_spots):
            d[k] = np.linalg.norm(center_image-centers[k, :])
        center_mole = centers[d.argmin(), :]

    # 6: take a subset of the image that includes the mole
    c0 = center_mole[0]
    c1 = center_mole[1]
    RR, CC = im_clust.shape
    stepmax = min([c0, RR-c0, c1, CC-c1])
    cond = True
    area_old = 0
    surf_old = 1
    step = 10  # each time the algorithm increases the area by 2*step pixels
    # horizontally and vertically
    im_sel = (im_clust == i_col)  # im_sel is a boolean NDarray with N1 row and N2 columns
    im_sel = im_sel*1  # im_sel is now an integer NDarray with N1 row and N2 columns
    while cond:
        subset = im_sel[c0-step:c0+step+1, c1-step:c1+step+1]
        area = np.sum(subset)
        Delta = np.size(subset)-surf_old
        surf_old = np.size(subset)
        if area > area_old+0.01*Delta:
            step = step+10
            area_old = area
            cond = True
            if step > stepmax:
                cond = False
        else:
            cond = False
            # subset is the serach area
    plt.matshow(subset)
    plt.title("Raw Binary Image")
    plt.savefig(file+"_raw.pdf")
    plt.show()

    row, col = np.shape(subset)

    # cleaning the photo [x]
    for i in range(row-1):
        for j in range(col-1):

            if i == 0 & j == 0:

                cnt = 3

                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == 0 & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row & j == 0:
                cnt = 3
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif j == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif j == row:
                cnt = 5

                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            else:
                cnt = 9

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt <= 3:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

    # cleaning the photo [y]
    for j in range(col-1):
        for i in range(row-1):

            if i == 0 & j == 0:

                cnt = 3

                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == 0 & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row & j == 0:
                cnt = 3
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif j == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif j == row:
                cnt = 5

                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            else:
                cnt = 9

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt <= 3:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

    count+=100

    # fill holes[1/4]
    for i in range(round(row/2))[::-1]:
        for j in range(round(col/2))[::-1]:

            cnt = 8
            if subset[i][j] == 1:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 0
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1

    # eliminate islands [1/4]
    for i in range(round(row/2)):
        for j in range(round(col/2)):

            cnt = 8
            if subset[i][j] == 0:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 1
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1

    # fill holes [2/4]
    for i in range(round(row/2)):
        for j in range(round(col/2), col-1)[::-1]:

            cnt = 8
            if subset[i][j] == 0:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 1
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1

    # eliminate islands [2/4]
    for i in range(round(row/2))[::-1]:
        for j in range(round(col/2), col-1):

            cnt = 8
            if subset[i][j] == 1:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 0
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1

    # fill holes [3/4]
    for i in range(round(row/2), row-1)[::-1]:
        for j in range(round(col/2)):

            cnt = 8
            if subset[i][j] == 0:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 1
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1

    # eliminate islands [3/4]
    for i in range(round(row/2), row-1):
        for j in range(round(col/2))[::-1]:

            cnt = 8
            if subset[i][j] == 1:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 0
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1

    # fill holes [4/4]
    for i in range(round(row/2), row-1)[::-1]:
        for j in range(round(col/2), col-1)[::-1]:

            cnt = 8
            if subset[i][j] == 0:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 1
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1
    # eliminate islands[4/4]
    for i in range(round(row/2), row-1):
        for j in range(round(col/2), col-1):

            cnt = 8
            if subset[i][j] == 1:
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1

                if cnt <= 4:

                    subset[i][j] = 0
                    plt.matshow(subset)
                    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                    count += 1

    count+=100

    # clean borders
    for i in range(row-1):
        subset[i][0] = 0
        plt.matshow(subset)
        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
        count += 1
    for i in range(col-1):
        subset[0][i] = 0
        plt.matshow(subset)
        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
        count += 1
    for i in range(row-1):
        subset[i][col-1] = 0
        plt.matshow(subset)
        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
        count += 1
    for i in range(col-1):
        subset[row-1][i] = 0
        plt.matshow(subset)
        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
        count += 1

    count+=100
    plt.matshow(subset)
    plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
    count += 1

    # final cleaning the photo [x]
    for i in range(row-1):
        for j in range(col-1):

            if i == 0 & j == 0:

                cnt = 3

                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == 0 & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row & j == 0:
                cnt = 3
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            elif i == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif j == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            elif j == row:
                cnt = 5

                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            else:
                cnt = 9

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt <= 4:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

    # final cleaning the photo [y]
    for j in range(col-1):
        for i in range(row-1):

            if i == 0 & j == 0:

                cnt = 3

                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == 0 & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif i == row & j == 0:
                cnt = 3
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            elif i == row & j == row:
                cnt = 3
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            elif i == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            elif i == row:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

            elif j == 0:
                cnt = 5

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            elif j == row:
                cnt = 5

                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt == 1:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1


            else:
                cnt = 9

                if subset[i][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i+1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j+1] != subset[i][j]:
                    cnt -= 1
                if subset[i-1][j-1] != subset[i][j]:
                    cnt -= 1
                if subset[i][j-1] != subset[i][j]:
                    cnt -= 1
                if cnt <= 4:

                    if subset[i][j] == 0:
                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
                    else:
                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

    count+=100
    # final retouch with higher resolution images
    if row >= 150:
        # fill holes[1/4]
        for i in range(round(row/2))[::-1]:
            for j in range(round(col/2))[::-1]:

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
        # eliminate islands [1/4]
        for i in range(round(row/2)):
            for j in range(round(col/2)):

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

        # fill holes [2/4]
        for i in range(round(row/2)):
            for j in range(round(col/2), col-1)[::-1]:

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
        # eliminate islands [2/4]
        for i in range(round(row/2))[::-1]:
            for j in range(round(col/2), col-1):

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

        # fill holes [3/4]
        for i in range(round(row/2), row-1)[::-1]:
            for j in range(round(col/2)):

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
        # eliminate islands [3/4]
        for i in range(round(row/2), row-1):
            for j in range(round(col/2))[::-1]:

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

        # fill holes [4/4]
        for i in range(round(row/2), row-1)[::-1]:
            for j in range(round(col/2), col-1)[::-1]:

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 1
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1
        # eliminate islands[4/4]
        for i in range(round(row/2), row-1):
            for j in range(round(col/2), col-1):

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i+1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j+1] != subset[i][j]:
                        cnt -= 1
                    if subset[i-1][j-1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j-1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:

                        subset[i][j] = 0
                        plt.matshow(subset)
                        plt.savefig('./moles/pdf/movie_'+str(count)+'.pdf')
                        count += 1

    # # cleaning hair [x-direct]
    # for i in range(row-1):
    #     flag = 0
    #     we = 0
    #     for j in range(col-1):
    #
    #         if subset[i][j] == 1 and flag == 0:
    #             flag = 1
    #
    #         if flag == 1 and subset[i][j] == 1:
    #             cnt += 1
    #
    #         if subset[i][j] == 0 and flag == 1 and cnt >= (round(row/4)):
    #
    #             for k in range(j, row):
    #
    #                 if subset[i][k] == 1:
    #                     we = k
    #                     break
    #
    #             for m in range(j, we):
    #                 subset[i][m] = 1
    #
    #             cnt = 0
    #             flag = 0
    #             i += 1
    #             j = 0
    #
    #         if subset[i][j] == 0 and flag == 1 and cnt < (round(row/4)):
    #             cnt = 0
    #             flag = 0
    #             i += 1
    #             j = 0
    #             break

    # # cleaning hair[y-direct]
    # for j in range(row-1):
    #     flag = 0
    #     we = 0
    #     for i in range(col-1):
    #
    #         if subset[i][j] == 1 and flag == 0:
    #             flag = 1
    #
    #         if flag == 1 and subset[i][j] == 1:
    #             cnt += 1
    #
    #         if subset[i][j] == 0 and flag == 1 and cnt >= (round(row/4)):
    #
    #             for k in range(j, row):
    #
    #                 if subset[i][k] == 1:
    #                     we = k
    #                     break
    #
    #             for m in range(j, we):
    #                 subset[i][m] = 1
    #
    #             cnt = 0
    #             flag = 0
    #             i += 1
    #             j = 0
    #
    #         if subset[i][j] == 0 and flag == 1 and cnt < (round(row/4)):
    #             cnt = 0
    #             flag = 0
    #             i += 1
    #             j = 0
    #             break

    # #cleaning hair [x-undirect]
    #     for i in range(row-1):
    #         flag = 0
    #         we=0
    #         for j in range(col-1)[::-1]:
    #
    #             if subset[i][j] == 1 and flag == 0:
    #                 flag = 1
    #
    #             if flag == 1 and subset[i][j] == 1:
    #                 cnt += 1
    #
    #             if subset[i][j] == 0 and flag == 1 and cnt >= (round(row/4)):
    #
    #                 for k in range(j, row)[::-1]:
    #
    #                     if subset[i][k] == 1:
    #                         we = k
    #                         break
    #
    #                 for m in range(j, we)[::-1]:
    #                     subset[i][m] = 1
    #
    #                 cnt = 0
    #                 flag = 0
    #                 i += 1
    #                 j = col
    #
    #             if subset[i][j] == 0 and flag == 1 and cnt < (round(row/4)):
    #                 cnt = 0
    #                 flag = 0
    #                 i += 1
    #                 j = col
    #                 break

    # perimeter
    perimeter = np.copy(subset)
    for i in range(row-1):
        for j in range(col-1):

            if subset[i][j] == 1 and subset[i-1][j] == 1 and subset[i+1][j] == 1 and subset[i][j-1] == 1 and subset[i][j+1] == 1:
                perimeter[i][j] = 0

    # plt.matshow(perimeter)
    # plt.title("Perimeter")
    # plt.savefig(file+"_perimeter.pdf")
    # plt.show()
    #
    # plt.matshow(subset)
    # plt.title("Area")
    # plt.savefig(file+"_polished.pdf")
    # plt.show()
