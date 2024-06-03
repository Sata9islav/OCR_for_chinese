import argparse
from collections import defaultdict
import cv2
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
import os
from paddleocr import PaddleOCR
import uvicorn

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='ch')


def average_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    return np.median(angles)


def rotate_to_horizontal(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    angle = average_angle(lines)
    if not np.isnan(angle):
        size_reverse = np.array(image.shape[1::-1]) # swap x with y
        M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
        MM = np.absolute(M[:,:2])
        size_new = MM @ size_reverse
        M[:,-1] += (size_new - size_reverse) / 2.
        return cv2.warpAffine(image, M, tuple(size_new.astype(int)))


def get_aug_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inverted_image = cv2.bitwise_not(thresholded_image)
    dilated_image = cv2.dilate(inverted_image,None, iterations=5)
    return dilated_image


def get_contour_of_table(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    contour_with_max_area = None
    for scale in np.linspace(0.005, 0.05, 10):
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            epsilon = scale * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    contour_with_max_area = approx
    return contour_with_max_area


def calculate_distance_between_2points(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def calculate_new_width_and_height_of_image(original_image, contour_with_max_area_ordered):
    existing_image_width = original_image.shape[1]
    existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
    distance_between_top_left_and_top_right = calculate_distance_between_2points(contour_with_max_area_ordered[0],
                                                                                 contour_with_max_area_ordered[1])
    distance_between_top_left_and_bottom_left = calculate_distance_between_2points(contour_with_max_area_ordered[0],
                                                                                   contour_with_max_area_ordered[3])
    aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right
    new_image_width = existing_image_width_reduced_by_10_percent
    new_image_height = int(new_image_width * aspect_ratio)
    return new_image_width, new_image_height


def apply_perspective_transform(new_image_width, new_image_height, contour_with_max_area_ordered, image):
    pts1 = np.float32(contour_with_max_area_ordered)
    pts2 = np.float32([[0, 0], [new_image_width, 0], [new_image_width, new_image_height], [0, new_image_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_corrected_image = cv2.warpPerspective(image, matrix, (new_image_width, new_image_height))
    return perspective_corrected_image


def add_padding_to_image(image):
    image_height = image.shape[0]
    padding = int(image_height * 0.05)
    perspective_corrected_image_with_padding = cv2.copyMakeBorder(image, padding, padding,
                                                                  padding, padding, cv2.BORDER_CONSTANT,
                                                                  value=[255, 255, 255])
    return perspective_corrected_image_with_padding


def get_rows_from_parse(result):
    list_with_lines = []
    for line in result[0]:
        list_with_lines.append(line)
    rows = defaultdict(list)

    for item in list_with_lines:
        block = item[0]
        average_y = sum(corner[1] for corner in block) / 4
        found_row = False
        for key in rows.keys():
            if abs(key - average_y) < 15:
                rows[key].append(item)
                found_row = True
                break
        if not found_row:
            rows[average_y] = [item]

    lst_with_rows = []
    sorted_rows = {i: rows[key] for i, key in enumerate(sorted(rows.keys()))}
    for row_index, blocks in sorted_rows.items():
        row_texts = [block[1][0] for block in sorted(blocks, key=lambda b: b[0][0][0])]
        lst_with_rows.append(f"Row {row_index + 1}: " + " ".join(row_texts))
    return lst_with_rows


def remove_file(file_path: str):
    os.remove(file_path)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/recognition")
async def get_form():
    return {"message": "Please use POST request to submit the file."}


@app.post("/recognition")
async def extract_text(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_file_path = f"tempfile_{file.filename}"
    contents = file.file.read()
    with open(temp_file_path, 'wb') as out_file:
        out_file.write(contents)

    image = cv2.imread(temp_file_path)
    rotated_image = rotate_to_horizontal(image)
    aug_image = get_aug_image(rotated_image)
    contour_with_max_area = get_contour_of_table(aug_image)
    contour_with_max_area_ordered = order_points(contour_with_max_area)
    new_image_width, new_image_height = calculate_new_width_and_height_of_image(rotated_image,
                                                                                contour_with_max_area_ordered)
    perspective_corrected_image = apply_perspective_transform(new_image_width,
                                                              new_image_height,
                                                              contour_with_max_area_ordered,
                                                              rotated_image)
    perspective_corrected_image_with_padding = add_padding_to_image(perspective_corrected_image)
    result = ocr.ocr(perspective_corrected_image_with_padding, cls=True)
    rows = get_rows_from_parse(result)
    background_tasks.add_task(remove_file, temp_file_path)
    return JSONResponse(status_code=200,
                        content={"message": f"Successfuly uploaded -> {file.filename}",
                                 "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8001, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    # parser.add_argument("--debug", action="store_true", dest="debug")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)
    uvicorn.run(app, host="0.0.0.0", port=8001, debug=True)
