from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import numpy as np
from tkinter.simpledialog import askfloat
from tkinter import messagebox,simpledialog,Label,Canvas
import os

panelA = None
panelA_image = None
panelB=None
image = None
path=None

def threshold():
    test_image = grayscale()

    ret, thresh = cv2.threshold(test_image, 127, 255, cv2.THRESH_BINARY)
    thresh1 = Image.fromarray(thresh)
    return thresh1
def convulate_2D(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')


    output_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            output_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output_image
def convulate_mode(image,kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    output_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            output_image[i, j] = find_mode(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output_image

def upload():
    global panelA,panelB, image,panelA_image
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png'),('TIF Files','*.Tif')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image = cv2.resize(image, (500, 500))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)

    image1 = ImageTk.PhotoImage(image1)

    panelA = Label(image=image1, borderwidth=5, relief="sunken")
    panelA.image = image1
    panelA.grid(row=1, column=1, rowspan=13, columnspan=3, padx=20, pady=20)
    panelA_image=image1
    return image
def grayscale():
    if image is not None:
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayimg1 = Image.fromarray(grayimg)
        grayimg1 = ImageTk.PhotoImage(grayimg1)
        panelB = Label(image=grayimg1, borderwidth=5, relief="sunken")
        panelB.image = grayimg1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return grayimg
def negative():
    if image is not None:
        neg = 255 - image
        neg1 = Image.fromarray(neg)
        neg1 = ImageTk.PhotoImage(neg1)
        panelB = Label(image=neg1, borderwidth=5, relief="sunken")
        panelB.image = neg1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return neg
def Binary():
    global image
    if image is not None:
        threshold_value = askfloat("threshold_value", "Enter the threshold_value (e.g.,0 to 255):", parent=root)
        if threshold_value is not None:
            threshold_value = max(0, min(255, threshold_value))
            ret, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            thresh_image = Image.fromarray(thresh)
            thresh_image_tk = ImageTk.PhotoImage(thresh_image)
            panelB = Label(image=thresh_image_tk, borderwidth=5, relief="sunken")
            panelB.image = thresh_image_tk
            panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
            return thresh
def create_threshold_slider():
    global threshold_var
    threshold_var = DoubleVar()
    threshold_var.set(128.0)  # Set an initial threshold value
def binary_threshold(threshold_value):
    global image, panelB
    if image is not None:
        # image=grayscale()
        ret, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        thresh_image = Image.fromarray(thresh)
        thresh_image_tk = ImageTk.PhotoImage(thresh_image)

        if panelB is not None:
            panelB.destroy()

        panelB = Label(image=thresh_image_tk, borderwidth=5, relief="sunken")
        panelB.image = thresh_image_tk
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
def edge():
    global image
    if image is not None:
        image = threshold()
        edged = cv2.Canny(image, 50, 100)
        edged1 = Image.fromarray(edged)
        edged1 = ImageTk.PhotoImage(edged1)
        panelB = Label(image=edged1, borderwidth=5, relief="sunken")
        panelB.image = edged1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return edged
def redext():
    global image
    if image is not None:
        blue, green, red = cv2.split(image)
        red_band = np.zeros_like(image)
        red_band[:, :, 2] = red
        red_band_image = Image.fromarray(cv2.cvtColor(red_band, cv2.COLOR_BGR2RGB))
        red_band_image_tk = ImageTk.PhotoImage(red_band_image)
        panelB = Label(image=red_band_image_tk, borderwidth=5, relief="sunken")
        panelB.image = red_band_image_tk
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return red_band
def greenext():
    global image
    if image is not None:
        blue, green, red = cv2.split(image)
        green_band = np.zeros_like(image)
        green_band[:, :, 1] = green
        green_band_image = Image.fromarray(cv2.cvtColor(green_band, cv2.COLOR_BGR2RGB))
        green_band_image_tk = ImageTk.PhotoImage(green_band_image)
        panelB = Label(image=green_band_image_tk, borderwidth=5, relief="sunken")
        panelB.image = green_band_image_tk
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return green_band
def blueext():
    global image
    if image is not None:
        row, col, plane = image.shape
        blue = np.zeros((row, col, plane), np.uint8)
        blue[:, :, 2] = image[:, :, 2]
        blue1 = Image.fromarray(blue)
        blue1 = ImageTk.PhotoImage(blue1)
        panelB = Label(image=blue1, borderwidth=5, relief="sunken")
        panelB.image = blue1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return blue
def contrast_enhancement():
    global image, panelB
    if image is not None:
        if image.dtype != np.uint8:
            return messagebox.showerror("Unsupported Image Format", "Image must be in 8-bit format (CV_8U)")
        x = askfloat("Enter min_intensity", "Enter valid number between 0 and 255", parent=root)
        y = askfloat("Enter max_intensity", "Enter valid number between 0 and 255", parent=root)
        if x is None or x > 255 or x < 0 or y is None or y > 255 or y < 0:
            return messagebox.showerror("Enter valid number between 0 and 255")
        min_intensity, max_intensity = x, y
        b, g, r = cv2.split(image)
        b_enhanced = cv2.convertScaleAbs(b, alpha=(max_intensity - min_intensity) / 255, beta=min_intensity)
        g_enhanced = cv2.convertScaleAbs(g, alpha=(max_intensity - min_intensity) / 255, beta=min_intensity)
        r_enhanced = cv2.convertScaleAbs(r, alpha=(max_intensity - min_intensity) / 255, beta=min_intensity)
        enhanced_image = cv2.merge([b_enhanced, g_enhanced, r_enhanced])
        contrast_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        contrast_image = ImageTk.PhotoImage(contrast_image)

        panelB = Label(image=contrast_image, borderwidth=5, relief="sunken")
        panelB.image = contrast_image
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return contrast_image
def skeleton():
    global image
    if image is not None:
        image = threshold()
        skel = np.zeros(image.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            open = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(image, open)
            eroded = cv2.erode(image, element)
            skel = cv2.bitwise_or(skel, temp)
            image = eroded.copy()
            if cv2.countNonZero(image) == 0:
                break
        skel1 = Image.fromarray(skel)
        skel1 = ImageTk.PhotoImage(skel1)
        panelB = Label(image=skel1, borderwidth=5, relief="sunken")
        panelB.image = skel1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return skel
def denoise():
    if image is not None:
        denoise = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        denoise1 = Image.fromarray(denoise)
        denoise1 = ImageTk.PhotoImage(denoise1)
        panelB = Label(image=denoise1, borderwidth=5, relief="sunken")
        panelB.image = denoise1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return denoise
# def generate_sharpening_kernel(entries,n):
#     if n % 2 == 0 or n < 3:
#         messagebox.showerror("Invalid Input", "n should be an odd number greater than or equal to 3.")
#         return None
#
#     kernel = np.zeros((n, n), dtype=int)
#
#     for i in range(n):
#         for j in range(n):
#                 value = int(entries[i][j].get())
#                 kernel[i, j] = value
#
#     return kernel
#
# def create_input_grid(window, n):
#     entries = []
#     for i in range(n):
#         row_entries = []
#         for j in range(n):
#             entry_var = tk.StringVar()
#             entry = tk.Entry(window, textvariable=entry_var, width=5, justify="center")
#             entry.grid(row=i, column=j, padx=2, pady=2)
#             row_entries.append(entry_var)
#
#         # Add a label for the row
#         label = tk.Label(window, text=f"Row {i + 1}", padx=5, pady=2, bg="pink")
#         label.grid(row=i, column=n, sticky="w")
#
#         entries.append(row_entries)
#
#     # Add labels for columns
#     for j in range(n):
#         label = tk.Label(window, text=f"Col {j + 1}", padx=5, pady=2, bg="pink")
#         label.grid(row=n, column=j)
#
#     # Add Submit button
#     submit_button = tk.Button(window, text="Submit", command=lambda : submit_values(entries))
#     submit_button.grid(row=n, column=n, padx=5, pady=5, sticky="e")
#
#     return entries
#
#
# def submit_values(entries):
#     values = []
#     for row in entries:
#         row_values = []
#         for entry_var in row:
#             value = entry_var.get()
#             row_values.append(value)
#         values.append(row_values)
#     return values
def sharp():
    global image,panelB
    if image is not None:
        # val=askinteger("odd value", "Enter the odd (e.g., 1,3,5,7):", parent=root)
        # entries=create_input_grid(root,val)
        # kernel=generate_sharpening_kernel(entries,val)
        # if kernel is None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0,-1,0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        sharpened1 = Image.fromarray(sharpened)
        sharpened1 = ImageTk.PhotoImage(sharpened1)
        panelB = Label(image=sharpened1, borderwidth=5, relief="sunken")
        panelB.image = sharpened1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return sharpened
def histo():
    if image is not None:
        histogram = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        histogram[:, :, 0] = cv2.equalizeHist(histogram[:, :, 0])
        histogram = cv2.cvtColor(histogram, cv2.COLOR_YUV2BGR)
        histogram1 = Image.fromarray(histogram)
        histogram1 = ImageTk.PhotoImage(histogram1)
        panelB = Label(image=histogram1, borderwidth=5, relief="sunken")
        panelB.image = histogram1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return histogram
def powerlawtrans():
    if image is not None:
        gamma=askfloat("Gamma","Enter a gamma value between 0 and 3")
        if gamma is not None:
            if gamma >=0 and gamma <=3:

                gammaplt=grayscale()
                gammaplt = np.array(255 * (gammaplt / 255) ** gamma, dtype='uint8')
                gammaplt1 = Image.fromarray(gammaplt)
                gammaplt1 = ImageTk.PhotoImage(gammaplt1)
                panelB = Label(image=gammaplt1, borderwidth=5, relief="sunken")
                panelB.image = gammaplt1
                panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
                return gammaplt
            else:
                return
def maskimg():
    if image is not None:
        x, y, w, h = cv2.selectROI(image)
        start = (x, y)
        end = (x + w, y + h)
        rect = (x, y, w, h)
        cv2.rectangle(image, start, end, (0, 0, 255), 3)
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        mask1 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        maskimage = image * mask1[:, :, np.newaxis]
        maskimage = Image.fromarray(maskimage)
        maskimage = ImageTk.PhotoImage(maskimage)
        # root=TK()
        # root.title("Mask Image")
        panelB = Label(image=maskimage, borderwidth=5, relief="sunken")
        panelB.image = maskimage
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return maskimage
def pencil():
    img = grayscale()
    if img is not None:
        img_invert = cv2.bitwise_not(img)
        img_smoothing = cv2.GaussianBlur(img_invert, (25, 25), sigmaX=0, sigmaY=0)
        pencilimg = cv2.divide(img, 255 - img_smoothing, scale=255)
        pencilimg1 = Image.fromarray(pencilimg)
        pencilimg1 = ImageTk.PhotoImage(pencilimg1)
        panelB = Label(image=pencilimg1, borderwidth=5, relief="sunken")
        panelB.image = pencilimg1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return pencilimg
def generate_gaussian_kernel(sigma,n):
    kernel = np.zeros((n, n), dtype=np.float32)
    k=(n-1)/2
    for i in range(n):
        for j in range(n):
            expo = -(((i - (k + 1)) ** 2 + (j - (k + 1)) ** 2) / (2 * sigma ** 2))
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(expo)
    kernel/=np.sum(kernel)
    return kernel
def generate_mean_kernel(n):
    kernel = np.ones((n, n), dtype=np.float32)
    kernel/=np.sum(kernel)
    return kernel
def gaussian_blur():
    global image,panelB
    if image is not None:
        x = askfloat("Enter Deviation", "Enter valid number between 0 and 10", parent=root)
        y = askfloat("Enter kernel size", "Enter valid number 3 or 5", parent=root)
        if x is not None and y is not None and x > 0 and y > 0:
            kernel = generate_gaussian_kernel(x, int(y))
            img = grayscale()
            # gauss = convulate_2D(img, kernel)
            gauss=cv2.filter2D(image,-1,kernel)
            gauss = Image.fromarray(gauss)
            gauss = ImageTk.PhotoImage(gauss)
            panelB = Label(image=gauss, borderwidth=5, relief="sunken")
            panelB.image = gauss
            panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
            return gauss
def mean_blur():
    global image
    if image is not None:
        x = askfloat("Enter KERNEL SIZE", "Enter 3 or 5", parent=root)
        if x is not None and x > 0 :
            kernel=generate_mean_kernel(int(x))
            # img=grayscale()
            # mean=convulate_2D(img, kernel)
            mean = cv2.filter2D(image,-1,kernel)
            mean = Image.fromarray(mean)
            mean = ImageTk.PhotoImage(mean)
            panelB = Label(image=mean,borderwidth=5, relief="sunken")
            panelB.image = mean
            panelB.grid(row=1, column=4, rowspan=13, columnspan=3,padx=20, pady=20)
            return mean
def median_blur():
    global panelB,image
    if image is not None:
        x = askfloat("Enter KERNEL SIZE", "Enter 3 or 5", parent=root)
        if x is not None and x > 0:
            kernel = np.ones((int(x),int(x)),np.uint8)
            median = medianblur(image,kernel)
            median = Image.fromarray(median)
            median = ImageTk.PhotoImage(median)
            panelB = Label(image=median,borderwidth=5, relief="sunken")
            panelB.image = median
            panelB.grid(row=1, column=4, rowspan=13, columnspan=3,padx=20, pady=20)
            return median

def find_median(matrix):
    flattened_array = np.ravel(matrix)
    sorted_array = np.sort(flattened_array)
    if len(sorted_array) % 2 == 1:
        median = sorted_array[len(sorted_array) // 2]
    else:
        middle1 = sorted_array[len(sorted_array) // 2 - 1]
        middle2 = sorted_array[len(sorted_array) // 2]
        median = (middle1 + middle2) / 2
    return median
def medianblur(image, kernel):
    image_height, image_width, channels = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Ensure that the padding dimensions are consistent with the image dimensions
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

    output_image = np.zeros_like(image, dtype=np.uint8)

    for c in range(channels):
        for i in range(image_height):
            for j in range(image_width):
                output_image[i, j, c] = find_median(padded_image[i:i + kernel_height, j:j + kernel_width, c] * kernel)

    return output_image
def sobel_filter(n):
    half_size = n // 2
    sobel_horizontal = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sobel_horizontal[i, j] = (i - half_size) * (j - half_size)
    sobel_vertical = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sobel_vertical[i, j] = (j - half_size) * (i - half_size)
    return sobel_horizontal,sobel_vertical
def sobel_edge(a):
    global image,panelB
    if image is not None:
        if a=="sobel_horizontal":
            kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        elif a=="sobel_vertical":
            kernel=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        elif a=="perwit_horizontal":
            kernel=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        elif a=="perwit_vertical":
            kernel=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        sobel=cv2.filter2D(image,-1,kernel)
        sobel=Image.fromarray(sobel)
        sobel=ImageTk.PhotoImage(sobel)
        panelB = Label(image=sobel, borderwidth=5, relief="sunken")
        panelB.image = sobel
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return sobel
def canny_edge(a):
    global image, panelB
    if image is not None:
        if a=="canny_edge":
            # image=gaussian_blur()
            # sobel_x = sobel_edge("sobel_horizontal")
            # sobel_y = sobel_edge("sobel_vertical")
            #
            # # Compute gradient magnitude and direction
            # gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            # gradient_direction = np.arctan2(sobel_y, sobel_x)
            #
            # # Step 3: Apply gradient magnitude thresholding
            # thresholded = np.zeros_like(gradient_magnitude)
            # thresholded[gradient_magnitude > 30] = 255  # Adjust the threshold as needed
            #
            # # Step 4: Apply double threshold
            # edges = np.zeros_like(thresholded)
            # edges[(thresholded >= low_threshold) & (thresholded <= high_threshold)] = 255
            #
            # # Step 5: Track edges by hysteresis
            # edges = cv2.Canny(blurred, low_threshold, high_threshold)
            # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            x = askfloat("Enter Gaussian KERNEL SIZE", "Enter 3 or 5", parent=root)
            q=int(x)
            blurred = cv2.GaussianBlur(image, (q, q), 0)

            # Step 2: Find intensity gradients of the image
            gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

            # Compute gradient magnitude and direction
            gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            # gradient_direction = np.arctan2(gradient_y, gradient_x)

            # Step 3: Apply gradient magnitude thresholding
            threshould=askfloat("Enter Threshould Value","Enter Threshould", parent=root)
            thresholded = np.zeros_like(gradient_magnitude)
            thresholded[gradient_magnitude > threshould] = 255  # Adjust the threshold as needed

            # Step 4: Apply double threshold
            edges = np.zeros_like(thresholded)
            low_threshold=askfloat("Enter Lower Threshold","Enter Lower Threshold", parent=root)
            high_threshold=askfloat("Enter Higher Threshold","Enter Higher Threshold", parent=root)
            edges[(thresholded >= low_threshold) & (thresholded <= high_threshold)] = 255

            # Step 5: Track edges by hysteresis
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            edges = Image.fromarray(edges)
            edges = ImageTk.PhotoImage(edges)
            panelB = Label(image=edges, borderwidth=5, relief="sunken")
            panelB.image = edges
            panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
            return edges
def sobel():
    global image
    if image is not None:
        # kernel=sobel_filter(3)
        # mode=convulate_mode(img,kernel)

        img=grayscale()
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        req = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        req = Image.fromarray(req)
        req = ImageTk.PhotoImage(req)
        panelB = Label(image=req,borderwidth=5,relief="sunken")
        panelB.image = req
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3,padx=20, pady=20)
        return req
def colpencil():
    if image is not None:
        img_invert = cv2.bitwise_not(image)
        img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
        colpencilimg = cv2.divide(image, 255 - img_smoothing, scale=255)
        colpencilimg1 = Image.fromarray(colpencilimg)
        colpencilimg1 = ImageTk.PhotoImage(colpencilimg1)
        panelB = Label(image=colpencilimg1, borderwidth=5, relief="sunken")
        panelB.image = colpencilimg1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return colpencilimg
def cartoon():
    gray = grayscale()
    if gray is not None:
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        cartoon1 = Image.fromarray(cartoon)
        cartoon1 = ImageTk.PhotoImage(cartoon1)
        panelB = Label(image=cartoon1, borderwidth=5, relief="sunken")
        panelB.image = cartoon1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return cartoon
def watercolor():
    if image is not None:
        watercolor = cv2.stylization(image, sigma_s=100, sigma_r=0.45)
        watercolor1 = Image.fromarray(watercolor)
        watercolor1 = ImageTk.PhotoImage(watercolor1)
        panelB = Label(image=watercolor1, borderwidth=5, relief="sunken")
        panelB.image = watercolor1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return watercolor
def emboss():
    if image is not None:
        kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
        emboss = cv2.filter2D(image, kernel=kernel, ddepth=-1)
        emboss = cv2.cvtColor(emboss, cv2.COLOR_BGR2GRAY)
        emboss = 255 - emboss
        emboss1 = Image.fromarray(emboss)
        emboss1 = ImageTk.PhotoImage(emboss1)
        panelB = Label(image=emboss1, borderwidth=5, relief="sunken")
        panelB.image = emboss1
        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        return emboss
def stretch_with_percentile():
    global image
    if image is not None:
        percentile = askfloat("Input Percentile", "Enter the percentile (e.g., 1 for 1%):", parent=root)
        if percentile is not None:
            if(percentile>100 or percentile<0):
                return messagebox.showerror("Enter valid percentile between 1 and 100")
            else:
                min_val, max_val = np.percentile(image, [percentile, 100 - percentile])
                stretched = np.clip((image - min_val) * (255.0 / (max_val - min_val)), 0, 255)
                stretched = ImageTk.PhotoImage(Image.fromarray(np.uint8(stretched)))
                panelB = Label(image=stretched, borderwidth=5, relief="sunken")
                panelB.image = stretched
                panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
                save_image()
                return stretched
def unsharp():
    global image
    if image is not None:
        img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel=generate_gaussian_kernel(0.5,3)
        blur_img=cv2.filter2D(img,-1,kernel)
        img_mask=cv2.subtract(img,blur_img)
        unsharp=cv2.addWeighted(img, 1, img_mask, -1, 0)
        unsharp=Image.fromarray(unsharp)
        unsharp=ImageTk.PhotoImage(unsharp)
        panelC=Label(image=unsharp, borderwidth=5,relief="sunken")
        panelC.image =unsharp
        panelC.grid(row=1, column=8, rowspan=13, columnspan=3, padx=20,pady=20)
        return unsharp
def lap():
    global image
    if image is not None:
        img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        lap_img=cv2.filter2D(img,-1,kernel)
        sharp_img=cv2.subtract(img,lap_img)
        sharp_img = Image.fromarray(sharp_img)
        sharp_img = ImageTk.PhotoImage(sharp_img)
        panelC = Label(image=sharp_img, borderwidth=5, relief="sunken")
        panelC.image = sharp_img
        panelC.grid(row=1, column=8, rowspan=13, columnspan=3, padx=20, pady=20)
        panele=Label()
        return sharp_img
# def Motion_Blur(image,Kernel_Size):
#     kernel=np.ones((Kernel_Size,Kernel_Size),np.float32)/Kernel_Size**2
#     img=cv2.filter2D(image,-1,kernel)
#     return img
# def add_poision_noise(image):
#     image = image.astype(np.float32)
#     scale=1.0
#     # Split the image into channels
#     b,g,r = cv2.split(image)
#
#     # Apply Poisson noise to each channel
#     b_image=np.random.poisson(b * scale)/scale
#     g_image=np.random.poisson(g * scale)/scale
#     r_image=np.random.poisson(r * scale)/scale
#
#     # Combine the noisy channels
#     noisy_image = cv2.merge((b_image,g_image,r_image))
#
#     # Clip values to the valid range [0, 255]
#     noisy_image = np.clip(noisy_image, 0, 255)
#
#     # Convert back to uint8 format
#     noisy_image = noisy_image.astype(np.uint8)
#
#     return noisy_image
# def add_gaussian_noise(image,mean,std):
#     # Generate Gaussian-distributed noise
#     noise = np.random.normal(mean, std, image.shape)
#
#     # Add noise to the image
#     noisy_image = image + noise
#
#     # Clip values to the valid range [0, 255]
#     noisy_image = np.clip(noisy_image, 0, 255)
#
#     # Convert back to uint8 format
#     noisy_image = noisy_image.astype(np.uint8)
#
#     return noisy_image
# def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
#     noisy_image = np.copy(image)
#
#     # Get the number of channels
#     num_channels = image.shape[2]
#
#     # Add salt and pepper noise to each channel individually
#     for i in range(num_channels):
#         # Add salt noise
#         salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
#         noisy_image[salt_mask, i] = 0
#
#         # Add pepper noise
#         pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
#         noisy_image[pepper_mask, i] = 255
#
#     return noisy_image
# def down_sample(image, factor):
#     fx = 1.0 / factor
#     fy = 1.0 / factor
#
#     # Perform downsampling using bicubic interpolation
#     downsampled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
#     downsampled_image=downsampled_image.astype(np.uint8)
#     return downsampled_image
# def up_sample(image,factor):
#     fx = factor
#     fy = factor
#     upsampled_image = cv2.resize(image, None, fx=fx, fy=fy,interpolation=cv2.INTER_CUBIC)
#     upsampled_image=upsampled_image.astype(np.uint8)
#     return upsampled_image
# def blur_gaussian(image, kernel_size, sigma):
#     # Apply Gaussian blur to each channel independently
#     blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
#     return blurred_image
# def downsample_bilinear(image, factor):
#     # Calculate the new dimensions
#     new_height = int(image.shape[0] / factor)
#     new_width = int(image.shape[1] / factor)
#
#     # Perform downsampling using bilinear interpolation for each channel
#     downsampled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
#     for i in range(image.shape[2]):
#         downsampled_image[:, :, i] = cv2.resize(image[:, :, i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#
#     return downsampled_image


# Motion Blur
def Motion_Blur(image, Kernel_Size):
    # kernel of size (Kernel_Size x Kernel_Size) with average value

    kernel = np.ones((Kernel_Size, Kernel_Size), np.float32) / np.sum(np.ones((Kernel_Size, Kernel_Size), np.float32))

    img = cv2.filter2D(image, -1, kernel)

    return img
# Gaussian Blur
def blur_gaussian(image, kernel_size, sigma):
    # Apply Gaussian blur to each channel independently

    blurred_image = np.zeros_like(image)

    for i in range(image.shape[2]):
        blurred_image[:, :, i] = cv2.GaussianBlur(image[:, :, i], (kernel_size, kernel_size), sigma)

    return blurred_image
# DOWN SAMPLING
def down_sample_bicubic(image, factor):
    fx = 1.0 / factor

    fy = 1.0 / factor

    # Perform downsampling using bicubic interpolation

    downsampled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    downsampled_image = downsampled_image.astype(np.uint8)

    return downsampled_image
# UP SAMPLING
def up_sample(image, factor):
    fx = factor

    fy = factor

    upsampled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    upsampled_image = upsampled_image.astype(np.uint8)

    return upsampled_image
def down_sample_bilinear(image, factor):
    # Calculate the new dimensions

    new_height = int(image.shape[0] / factor)

    new_width = int(image.shape[1] / factor)

    # Perform downsampling using bilinear interpolation for each channel

    downsampled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    for i in range(image.shape[2]):
        downsampled_image[:, :, i] = cv2.resize(image[:, :, i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return downsampled_image
def downsample_nearest_neighbor(image, factor):
    # Calculate the new dimensions

    new_height = int(image.shape[0] / factor)

    new_width = int(image.shape[1] / factor)

    # Perform nearest-neighbor downsampling

    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    return downsampled_image
# GAUSSIAN NOISE
def add_gaussian_noise(image, mean, std):
    # Generate Gaussian-distributed noise

    noise = np.random.normal(mean, std, image.shape)

    # Add noise to the image

    noisy_image = image + noise

    # Clip values to the valid range [0, 255]

    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert back to uint8 format

    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image
# POISSION NOISE
def add_poision_noise(image, scale=1.0):
    image = image.astype(np.float32)

    # Split the image into channels

    b, g, r = cv2.split(image)

    # Apply Poisson noise to each channel

    b_image = np.random.poisson(b * scale) / scale

    g_image = np.random.poisson(g * scale) / scale

    r_image = np.random.poisson(r * scale) / scale

    # Combine the noisy channels

    noisy_image = cv2.merge((b_image, g_image, r_image))

    # Clip values to the valid range [0, 255]

    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert back to uint8 format

    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image
# SALT AND PEPPER NOISE
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)

    # Get the number of channels

    num_channels = image.shape[2]

    # Add salt and pepper noise to each channel individually

    for i in range(num_channels):
        # Add salt noise

        salt_mask = np.random.rand(*image.shape[:2]) < salt_prob

        noisy_image[salt_mask, i] = 0

        # Add pepper noise

        pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob

        noisy_image[pepper_mask, i] = 255

    return noisy_image
# JPEG NOIS
def add_jpeg_noise(img, quality=95):
    # Decompress to obtain raw pixel values

    _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    img_decoded = cv2.imdecode(img_encoded, 1)

    # Add noise to each color channel

    noise = np.random.normal(0, 25, img_decoded.shape)

    noisy_image = np.clip(img_decoded + noise, 0, 255).astype(np.uint8)

    return noisy_image
def path1():
    global image
    if image is not None:
            test_image_path1 = image

            # Motion Blur -- parameters required image and kernel size

            motion_blur_image_path1 = Motion_Blur(test_image_path1, 3)

            # Down sampling by bicubic --  parameters required image and factor  --  here factor is 2 simply -- D2

            down_sampled_image_path1 = down_sample_bicubic(motion_blur_image_path1, 2)

            # Adding Poission Noise --  parameters required image --  here scale taken as

            poission_noise_added_image_path1 = add_poision_noise(down_sampled_image_path1)

            # Adding JPEG Noise -- parameters required image -- quality taken as 95

            jpeg_noise_added_image_path1 = add_jpeg_noise(poission_noise_added_image_path1)

            Final_Image = Image.fromarray(jpeg_noise_added_image_path1)

            panelB = Label(borderwidth=5, relief="sunken")

            Final_Image = ImageTk.PhotoImage(Final_Image)
            for widget in panelB.winfo_children():
                widget.destroy()

            # Final_Image = ImageTk.PhotoImage(Final_Image)

            panelB.configure(image=Final_Image)

            panelB.image = Final_Image

            panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)

            return Final_Image
def path2():
    global image
    if image is not None:
        test_image_path2 = image

        # Down sampling by bicubic --  parameters required image and factor  --  here factor is 2 simply -- D2

        down_sampled_image_path2 = down_sample_bicubic(test_image_path2, 2)

        # Adding Gaussian Noise --  parameters required image , mean , standard deviation

        gaussiann_noise_added_image_path2 = add_gaussian_noise(down_sampled_image_path2, 0, 1)

        # Gaussian Blurring --  parameters required image , kernel size , standard deviation

        gaussiann_blur_image_path2 = blur_gaussian(gaussiann_noise_added_image_path2, 3, 1)

        # Down Sampling by Bilinear --  parameters required image , factor -- here factor is 1 simply -- D bilinear

        down_sampled_image_bilinear_path2 = down_sample_bilinear(gaussiann_blur_image_path2, 1)

        # Adding JPEG Noise -- parameters required image -- quality taken as 95

        jpeg_noise_added_image_path2 = add_jpeg_noise(down_sampled_image_bilinear_path2)

        Final_Image = Image.fromarray(jpeg_noise_added_image_path2)

        panelB = Label(borderwidth=5, relief="sunken")

        Final_Image = ImageTk.PhotoImage(Final_Image)

        for widget in panelB.winfo_children():
            widget.destroy()

        panelB.configure(image=Final_Image)

        panelB.image = Final_Image

        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)

        return Final_Image
def path3():
    global image
    if image is not None:
        test_image_path3 = image

        # Down sampling by bicubic --  parameters required image and factor  --  here factor is 4 simply -- D4

        down_sampled_image_path3 = down_sample_bicubic(test_image_path3, 4)

        # Motion Blur -- parameters required image and kernel size

        motion_blur_image_path3 = Motion_Blur(down_sampled_image_path3, 3)

        # Up sampling by bicubic --  parameters required image and factor  --  here factor is 2  simply D2 up

        up_sampled_image_path3 = up_sample(motion_blur_image_path3, 2)

        # Adding Gaussian Noise --  parameters required image , mean , standard deviation

        gaussian_noise_added_image_path3 = add_gaussian_noise(up_sampled_image_path3, 0, 1)

        # Adding JPEG Noise -- parameters required image -- quality taken as 95

        jpeg_noise_added_image_path3 = add_jpeg_noise(gaussian_noise_added_image_path3)

        Final_Image = Image.fromarray(jpeg_noise_added_image_path3)

        Final_Image = ImageTk.PhotoImage(Final_Image)

        panelB = Label(borderwidth=5, relief="sunken")


        for widget in panelB.winfo_children():
            widget.destroy()
        panelB.configure(image=Final_Image)

        panelB.image = Final_Image

        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)

        return Final_Image
def path4():
    global image,panelB
    if image is not None:
        test_image_path4 = image

        # Adding Gaussian Noise --  parameters required image , mean , standard deviation

        gaussiann_noise_added_image_path4 = add_gaussian_noise(test_image_path4, 0, 1)

        # Down sampling by bicubic --  parameters required image and factor  --  here factor is 4 simply -- D4

        down_sampled_image_path4 = down_sample_bicubic(gaussiann_noise_added_image_path4, 4)

        # Down sampling by nearest neighbour -- parameters required image , factor -- here factor is 2 simply -- D2

        down_sampled_neighbour_image_path4 = downsample_nearest_neighbor(down_sampled_image_path4, 2)

        # Adding salt and pepper Noise --  parameters required image , salt_probability , pepper_probability -- here both are 0.01 by random

        salt_pepper_noise_added_image_path4 = add_salt_pepper_noise(down_sampled_neighbour_image_path4, salt_prob=0.01,
                                                                    pepper_prob=0.01)

        # Adding JPEG Noise -- parameters required image -- quality taken as 95

        jpeg_noise_added_image_path4 = add_jpeg_noise(salt_pepper_noise_added_image_path4)

        Final_Image = Image.fromarray(jpeg_noise_added_image_path4)

        panelB = Label(borderwidth=5, relief="sunken")

        Final_Image = ImageTk.PhotoImage(Final_Image)
        for widget in panelB.winfo_children():
            widget.destroy()

        panelB.configure(image=Final_Image)

        panelB.image = Final_Image

        panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)

        return Final_Image
def generate_gaussian_kernel(sigma,n):
    kernel = np.zeros((n, n), dtype=np.float32)
    k=(n-1)/2
    for i in range(n):
        for j in range(n):
            expo = -(((i - (k + 1)) ** 2 + (j - (k + 1)) ** 2) / (2 * sigma ** 2))
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(expo)
    kernel/=np.sum(kernel)
    return kernel
def save_image():
    global image,panelB
    if image is not None and panelB is not None:
        # if panelB.image is not None:
        # pil_image = Image.fromarray(panelB.image)
        # np_array=np.array(pil_image)
        # pil_image = Image.new("RGB", (1, 1))
        # pil_image.putdata(panelB.image.getdata())
        destination_folder = r"C:\Users\Training\Desktop\saved"
        os.makedirs(destination_folder, exist_ok=True)
        name = simpledialog.askstring("Name", "Enter your file name", parent=root)
        filename = os.path.join(destination_folder, f"{name}.tif")
        # cv2.imwrite(filename, cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR))
        # pil_image.save(filename)
        canvas = Canvas(root, width=panelB.image.width(), height=panelB.image.height())
        canvas.create_image(0, 0, anchor="nw", image=panelB.image)
        canvas.postscript(file=filename, colormode="color")
        messagebox.showinfo("Image saved", f"Image saved successfully as {filename}")
def apply_blur(selected_blur):
    if selected_blur == "Gaussian Blur":
        gaussian_blur()
    elif selected_blur == "Mean Blur":
        mean_blur()
    elif selected_blur == "Median":
        median_blur()
def apply_band(selected_band):
    if selected_band == "Red Attribute":
        redext()
    elif selected_band == "Green Attribute":
        greenext()
    elif selected_band == "Blue Attribute":
        blueext()
def apply_contrast(selected_contrast):
    if selected_contrast == "contrast enhancement":
        contrast_enhancement()
    elif selected_contrast == "power law transformation":
        powerlawtrans()
    elif selected_contrast == "percentile streching":
        stretch_with_percentile()
def apply_Edge(selected_Edge):
    if selected_Edge=="sobel_horizontal":
        sobel_edge("sobel_horizontal")
    elif selected_Edge=="sobel_vertical":
        sobel_edge("sobel_vertical")
    elif selected_Edge=="perwit_horizontal":
        sobel_edge("perwit_horizontal")
    elif selected_Edge=="perwit_vertical":
        sobel_edge("perwit_vertical")
    elif selected_Edge=="canny_edge":
        canny_edge("canny_edge")
def apply_sharp(selected_sharp):
    if selected_sharp == "Unsharp Masking":
        unsharp()
    if selected_sharp == "laplacian_sharpening":
        lap()
root = Tk()
root.title("IMAGE PROCESSING")


l1= Label(root, text="CLICK THE BUTTONS TO PERFORM THE FUNCTIONALITIES MENTIONED",
           fg="white", bg="#1d2951", width= 98, borderwidth=5, relief="groove",  font =('Verdana', 15))
l1.grid(row= 0, column= 0, columnspan= 6, padx=20, pady=20, sticky='nesw')

band_options = ["Select band Type", "Red Attribute", "Green Attribute", "Blue Attribute"]
selected_band_option = StringVar(root)
selected_band_option.set(band_options[0])  # Set the default option
band_dropdown = OptionMenu(root, selected_band_option, *band_options, command=apply_band)
band_dropdown.config(fg="white", bg="#1d2951")
band_dropdown.grid(row=1, column=1, padx=10, pady=10, sticky='nesw')




Edge_options = ["Select Edge Type","sobel_horizontal","sobel_vertical","perwit_horizontal","perwit_vertical","canny_edge"]
selected_Edge_option = StringVar(root)
selected_Edge_option.set(Edge_options[0])  # Set the default option
Edge_dropdown = OptionMenu(root, selected_Edge_option, *Edge_options, command=apply_Edge)
Edge_dropdown.config(fg="white", bg="#1d2951")
Edge_dropdown.grid(row=1, column=5, padx=10, pady=10, sticky='nesw')




blur_options = ["Select Blur Type", "Gaussian Blur", "Mean Blur", "Median"]
selected_blur_option = StringVar(root)
selected_blur_option.set(blur_options[0])  # Set the default option
blur_dropdown = OptionMenu(root, selected_blur_option, *blur_options, command=apply_blur)
blur_dropdown.config(fg="white", bg="#1d2951")
blur_dropdown.grid(row=1, column=2, padx=10, pady=10, sticky='nesw')




contrast_options = ["Select contrast Type", "contrast enhancement", "power law transformation", "percentile streching"]
selected_contrast_option = StringVar(root)
selected_contrast_option.set(contrast_options[0])  # Set the default option
contrast_dropdown = OptionMenu(root, selected_contrast_option, *contrast_options, command=apply_contrast)
contrast_dropdown.config(fg="white", bg="#1d2951")
contrast_dropdown.grid(row=1, column=3, padx=10, pady=10, sticky='nesw')



sharp_options = ["Select sharpening Type", "laplacian_sharpening","Unsharp Masking"]
selected_sharp_option = StringVar(root)
selected_sharp_option.set(sharp_options[0])  # Set the default option
sharp_dropdown = OptionMenu(root, selected_sharp_option, *sharp_options, command=apply_sharp)
sharp_dropdown.config(fg="white", bg="#1d2951")
sharp_dropdown.grid(row=1, column=4, padx=10, pady=10, sticky='nesw')


btn= Button(root, text="UPLOAD", fg="white", bg="#1d2951", command=upload)
btn.grid(row= 1, column= 0, padx=10, pady=10, sticky='nesw')

btn1= Button(root, text="GRAYSCALE", fg="white", bg="snow4", command=grayscale)
btn1.grid(row= 2, column= 0, padx=10, pady=10, sticky='nesw')

btn2= Button(root, text="INVERT COLOR", fg="white", bg="black", command=negative)
btn2.grid(row= 3, column= 0, padx=10, pady=10, sticky='nesw')


btn6= Button(root, text="BINARY", fg="white", bg="black", command=Binary)
btn6.grid(row= 4, column= 0, padx=10, pady=10, sticky='nesw')

btn7= Button(root, text="EDGE DETECTION", fg="white", bg="black", command=edge)
btn7.grid(row= 5, column= 0, padx=10, pady=10, sticky='nesw')

btn8= Button(root, text="SKELETON", fg="white", bg="black", command=skeleton)
btn8.grid(row= 6, column= 0, padx=10, pady=10, sticky='nesw')


btn11= Button(root, text="Histogram equalization", fg="white", bg="#1d2951",command=histo)
btn11.grid(row=7, column= 0, padx=10,pady=10,sticky='nesw')

btn12= Button(root, text="SHARPENING", fg="white", bg="#1d2951", command=sharp)
btn12.grid(row=8, column= 0, padx=10, pady=10, sticky='nesw')

btn13= Button(root, text="SMOOTHENING", fg="white", bg="#1d2951", command=denoise)
btn13.grid(row= 9, column= 0, padx=10, pady=10, sticky='nesw')

btn14= Button(root, text="REMOVE BACKGROUND", fg="white", bg="#1d2951", command=maskimg)
btn14.grid(row= 10, column= 0, padx=10, pady=10, sticky='nesw')

btn15= Button(root, text="PENCIL SKETCH", fg="white", bg="#1d2951", command=pencil)
btn15.grid(row= 11, column= 0, padx=10, pady=10, sticky='nesw')


btn16= Button(root, text="COLOR PENCIL SKETCH", fg="white", bg="#1d2951", command=colpencil)
btn16.grid(row= 12, column= 0, padx=10, pady=10, sticky='nesw')

btn17= Button(root, text="CARTOONIFY", fg="white", bg="#1d2951", command=cartoon)
btn17.grid(row= 13, column= 0, padx=10, pady=10, sticky='nesw')

btn18= Button(root, text="WATERCOLOR", fg="white", bg="#1d2951", command=watercolor)
btn18.grid(row= 14, column= 0, padx=10, pady=10, sticky='nesw')

btn19= Button(root, text="EMBOSS", fg="white", bg="#1d2951", command=emboss)
btn19.grid(row= 15, column= 0, padx=10, pady=10, sticky='nesw')


btn21 = Button(root, text="Save Image", fg="white", bg="green", command=save_image)
btn21.grid(row=16, column=0, padx=10, pady=10, sticky='nesw')

btn_path1=Button(root, text="path1", fg="white", bg="#1d2951",command=path1)
btn_path1.grid(row=16, column=1, padx=10, pady=10,sticky='nesw')

btn_path2=Button(root, text="path2", fg="white", bg="#1d2951",command=path2)
btn_path2.grid(row=16, column=2, padx=10, pady=10,sticky='nesw')

btn_path3=Button(root, text="path3", fg="white", bg="#1d2951",command=path3)
btn_path3.grid(row=16, column=3, padx=10, pady=10,sticky='nesw')

btn_path4=Button(root, text="path4", fg="white", bg="#1d2951",command=path4)
btn_path4.grid(row=16, column=4, padx=10, pady=10,sticky='nesw')




root.mainloop()


