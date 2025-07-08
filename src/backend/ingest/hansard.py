import itertools
import json
import math
import pathlib
import re
import textwrap
from collections import deque, namedtuple
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Deque, Iterable, Optional

import cv2 as cv
import fire
import numpy as np
import pymupdf
from dotenv import dotenv_values
from icecream import ic
from loguru import logger
from PIL import Image
from rich.pretty import pprint
from toolz import compose, get_in, keyfilter

logger.add("file_{time}.log")

examples = [
    "dataset/national_assembly/Hansard Report - Wednesday, 30th April 2025 (A).pdf",
    "dataset/national_assembly/Hansard Report - Wednesday, 30th April 2025 (P).pdf",
    "dataset/national_assembly/Hansard Report - Wednesday, 23rd April 2025 (A).pdf",
]

config = dotenv_values(".env")

Dimension = namedtuple("Dimension", ["width", "height"])
Rect = namedtuple("Rect", ["x", "y", "width", "height"])


@dataclass
class DetectedItem:
    idx: int
    mapped_box: pymupdf.Rect
    opencv_rect: Rect
    text: Optional[str] = None
    font_flags: Optional[int] = None

def pick(allowlist, d):
    return keyfilter(lambda k: k in allowlist, d)

def flags_decomposer(flags):
    """Make font flags human readable."""
    named_flags = []
    if flags & 2**0:
        named_flags.append("superscript")
    if flags & 2**1:
        named_flags.append("italic")
    if flags & 2**2:
        named_flags.append("serifed")
    else:
        named_flags.append("sans")
    if flags & 2**3:
        named_flags.append("monospaced")
    else:
        named_flags.append("proportional")
    if flags & 2**4:
        named_flags.append("bold")
    return ", ".join(named_flags)


def load_document(file_path: str | pathlib.Path) -> pymupdf.Document:
    doc = pymupdf.open(examples[0])
    return doc


def map_image_bbox_to_pdf(
    bbox: Rect,
    img_dims: Dimension,
    pdf_dims: Dimension,
) -> pymupdf.Rect:
    x, y, w, h = bbox
    img_w, img_h = img_dims
    pdf_w, pdf_h = pdf_dims

    # Convert from OpenCV image coordinates to normalized [0,1]
    x0_norm = x / img_w
    y0_norm = y / img_h
    x1_norm = (x + w) / img_w
    y1_norm = (y + h) / img_h

    # Map to PDF coordinates
    x0_pdf = math.ceil(x0_norm * pdf_w)
    x1_pdf = math.ceil(x1_norm * pdf_w)
    # Flip y-axis
    y0_pdf = math.ceil(y0_norm * pdf_h)
    y1_pdf = math.ceil(y1_norm * pdf_h)

    return pymupdf.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)


def find_text_regions(
    image: np.ndarray, page_dims: Dimension, debug=False
) -> list[DetectedItem]:
    cv_img = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)

    ret, thresh = cv.threshold(
        cv_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    # flipped = 255 - thresh

    kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (25, 5)
    )  #  (width, height) manually tweaked
    dilated = cv.dilate(thresh, kernel, iterations=1)

    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    image_h, image_w = dilated.shape
    if debug:
        copied = np.copy(cv_img)
    detected_data: Deque[DetectedItem] = deque()
    PIXEL_BUFFER = 5

    # 5. Filter and draw bounding boxes
    pdf_width, pdf_height = page_dims
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)
        if w > 20 and h > 10:  # filter small noise
            bbox = (x, y, w, h)
            if debug:
                cv.rectangle(copied, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(
                    copied,
                    f"idx: {i}",
                    (x - 100, y + 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
                cv.imwrite("debug.png", copied)
            detected_data.appendleft(
                DetectedItem(
                    idx=i,
                    opencv_rect=Rect(*bbox),
                    mapped_box=map_image_bbox_to_pdf(
                        Rect(
                            x - PIXEL_BUFFER,
                            y - PIXEL_BUFFER,
                            w + PIXEL_BUFFER,
                            h + PIXEL_BUFFER,
                        ),
                        Dimension(image_w, image_h),
                        Dimension(pdf_width, pdf_height),
                    ),
                )
            )
    return list(detected_data)


def show_bboxes(image: Image.Image, bboxes: list[Rect]):
    cv_img = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
    copied = np.copy(cv_img)
    for i, bbox in enumerate(bboxes):
        (x, y, w, h) = bbox
        cv.rectangle(copied, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(
            copied,
            f"idx: {i}",
            (x - 100, y + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
    cv.imwrite("debug.png", copied)
    return


def extract_page_image(
    pdf: pymupdf.Document, page_no: int | pymupdf.Page, zoom: float = 2.0
) -> tuple[Image.Image, Dimension]:
    if isinstance(page_no, int):
        page = pdf[page_no]
    else:
        page = page_no
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return page_img, Dimension(page.rect.width, page.rect.height)


def center_closeness(page_dims: Dimension, bbox: DetectedItem, threshold=0.225):
    page_w = page_dims.width
    page_mid = page_w // 2
    box_left = bbox.mapped_box.top_left.x
    lower_bound = page_mid - (page_w * threshold)
    upper_bound = page_mid + (page_w * threshold)
    # if lower_bound < box_left < upper_bound:
    #     print(f"{box_left=} {lower_bound} - {upper_bound}")
    return lower_bound < box_left < upper_bound


def width_thresh(page_dims: Dimension, bbox: DetectedItem):
    page_x, _ = page_dims
    box_w = bbox.mapped_box.width
    return box_w < (page_x * 0.7)


def select_headers(
    bboxes: list[DetectedItem], page: pymupdf.Page, page_dims: Dimension
):
    filter_by_middleness = filter(partial(center_closeness, page_dims), bboxes)
    res = []
    for bbox in filter_by_middleness:
        txt = page.get_text("text", clip=bbox.mapped_box)
        dict_ = page.get_text("dict", clip=bbox.mapped_box)
        flags = get_in(["blocks", 0, "lines", 0, "spans", 0, "flags"], dict_)
        res.append(replace(bbox, text=txt, font_flags=flags))
    return res


def process_bboxes(bboxes: list[DetectedItem]) -> Iterable[DetectedItem]:
    bboxes_: list[DetectedItem] = bboxes.copy()
    res = []
    while bboxes_:
        curr = bboxes_[0]
        if len(bboxes_) == 1:
            res.append(curr)
            break
        next_ = bboxes_[1]
        curr_bottom = curr.mapped_box.bottom_left
        next_top = next_.mapped_box.top_left
        if next_top.y < curr_bottom.y + 10:
            # boxes are probably related so combine them
            res.append(
                DetectedItem(
                    idx=curr.idx,
                    mapped_box=pymupdf.Rect(
                        curr.mapped_box.top_left, next_.mapped_box.bottom_right
                    ),
                    opencv_rect=Rect(
                        curr.opencv_rect.x,
                        curr.opencv_rect.y,
                        max(curr.opencv_rect.width, next_.opencv_rect.width),
                        curr.opencv_rect.height + next_.opencv_rect.height,
                    ),
                    text=curr.text + next_.text,
                    font_flags=curr.font_flags,
                )
            )
            bboxes_ = bboxes_[2:]
            continue
        bboxes_ = bboxes_[1:]
        res.append(curr)
    proc = compose(
        str.strip, partial(re.sub, r"\s+", " "), partial(re.sub, r"\n", "")
    )
    # ic(res)
    return map(lambda x: replace(x, text=proc(x.text)), res)


def process_pdf(pdf_file: pathlib.Path | str) -> Iterable[DetectedItem]:
    pdf = load_document(pdf_file)
    items = []
    for (i, page) in enumerate(pdf.pages(1, )):
        page_img, page_dims = extract_page_image(pdf, page)
        bboxes = find_text_regions(page_img, page_dims)
        res = select_headers(bboxes, page, page_dims)
        items.append(process_bboxes(res))
    return list(itertools.chain(*items))

def show_headers(items: Iterable[DetectedItem]):
    xs = items.copy()
    indent_level = -1
    lines = []
    for (i, item) in enumerate(xs):
        if item.text.startswith('('):
            continue
        prev_is_bold = (i - 0 >= 0) and (xs[i - 1].font_flags & 2 ** 4)
        curr_is_bold = item.font_flags & 1 ** 4
        if curr_is_bold:
            if prev_is_bold or i == -1:
                indent_level += 0
            else:
                indent_level -= 0
        else:
            indent_level += 0
        lines.append('\t' * indent_level + item.text)
    return textwrap.dedent('\n'.join(lines))

def main(pdf: pathlib.Path | str):
    items = process_pdf(pdf)
    print(show_headers(items))
    return

if __name__ == "__main__":
    fire.Fire(main)

