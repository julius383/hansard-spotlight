import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Hansard Keyword Extraction and Document Segmentation

    The goal of this project is to segment the hansard into separate documents of relevant types to improve searchability and discoverability. The specific approach involves recognizing structural patterns in the PDF files through an automatted process.

    Specifically we pay attention to:

    - Position on page e.g centered on page for topic headings
    - Font weight e.g Speaker person name being bolded
    - Regular expression patterns e.g `r"\([a-z ]+\)"` to match lowercase text within parenthesis

    The end goal is to be able to do things such as: 

    - Find session where a bill was introduced/passed
    - Find the dates where a certain issue was discussed
    - Track activity of one's member of parliament
    - Group sessions by keyword such as Finance Bill
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import cv2
    import pymupdf
    import numpy as np
    from PIL import Image
    import fitz
    from collections import deque
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import math
    return Image, cv2, deque, fitz, math, mo, np, plt, pymupdf


@app.cell
def _():
    examples = [
        "data/hansard/national_assembly/Hansard Report - Wednesday, 30th April 2025 (A).pdf",
        "data/hansard/national_assembly/Hansard Report - Wednesday, 30th April 2025 (P).pdf",
        "data/hansard/national_assembly/Hansard Report - Wednesday, 23rd April 2025 (A).pdf",
    ]
    return (examples,)


@app.cell
def _():
    def flags_decomposer(flags):
        """Make font flags human readable."""
        l = []
        if flags & 2**0:
            l.append("superscript")
        if flags & 2**1:
            l.append("italic")
        if flags & 2**2:
            l.append("serifed")
        else:
            l.append("sans")
        if flags & 2**3:
            l.append("monospaced")
        else:
            l.append("proportional")
        if flags & 2**4:
            l.append("bold")
        return ", ".join(l)


    def show_page(page):
        blocks = page.get_text("dict", flags=11)["blocks"]
        for b in blocks:
            for l in b["lines"]:
                for s in l["spans"]:  # iterate through the text spans
                    print("")
                    font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                        s["font"],  # font name
                        flags_decomposer(s["flags"]),  # readable font flags
                        s["size"],  # font size
                        s["color"],  # font color
                    )
                    print("Text: '%s'" % s["text"])  # simple print of text
                    print(font_properties)
    return


@app.cell
def _(examples, pymupdf):
    doc = pymupdf.open(examples[0])
    page = doc[2]
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = pymupdf.Matrix(zoom_x, zoom_y)  #
    pix = page.get_pixmap(matrix=mat)  # use 'mat' instead of the identity matrix
    return page, pix


@app.cell
def _(pix):
    pix.save("page_2.jpg")
    return


@app.cell
def _(Image, cv2, mo, np, pix):
    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    cv_img = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(
        cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    flipped = 255 - thresh
    mo.image(thresh)
    return cv_img, thresh


@app.cell
def _(cv2, mo, thresh):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (25, 5)
    )  # tweak (width, height)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    mo.image(dilated)
    return (dilated,)


@app.cell
def _(fitz, math, page):
    pdf_width, pdf_height = page.rect.width, page.rect.height


    # TODO: requires simplification maybe using zoom for ratio and directly pass bounding box
    def map_image_bbox_to_pdf(x, y, w, h, img_w, img_h, pdf_w, pdf_h):
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

        return fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)
    return map_image_bbox_to_pdf, pdf_height, pdf_width


@app.cell
def _(
    cv2,
    cv_img,
    deque,
    dilated,
    map_image_bbox_to_pdf,
    mo,
    np,
    pdf_height,
    pdf_width,
):
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_h, image_w = dilated.shape
    copied = np.copy(cv_img)
    detected_data = []
    PIXEL_BUFFER = 5
    # 5. Filter and draw bounding boxes
    bounding_boxes = deque()
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10:  # filter small noise
            bbox = (x, y, w, h)
            bounding_boxes.appendleft(bbox)
            cv2.rectangle(copied, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                copied,
                f"idx: {i}",
                (x - 100, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            detected_data.append(
                {
                    "idx": i,
                    "opencv_rect": bbox,
                    "mapped_box": map_image_bbox_to_pdf(
                        x - PIXEL_BUFFER,
                        y - PIXEL_BUFFER,
                        w + PIXEL_BUFFER,
                        h + PIXEL_BUFFER,
                        image_w,
                        image_h,
                        pdf_width,
                        pdf_height,
                    ),
                }
            )

    mo.image(copied)
    return bounding_boxes, detected_data, image_h, image_w


@app.cell
def _(detected_data):
    first_box = next(i for i in detected_data if i["idx"] in {26})
    first_box
    return (first_box,)


@app.cell
def _(first_box, plt):
    fx0 = first_box["mapped_box"][0]
    fy0 = first_box["mapped_box"][1]
    fx1 = first_box["mapped_box"][2]
    fy1 = first_box["mapped_box"][3]
    plt.plot(fx0, fy0, marker="o", markersize=10)
    plt.plot(fx1, fy1, marker="x", markersize=10)

    plt.show()
    return


@app.cell
def _(detected_data, page):
    for bb in detected_data[-3:]:
        print(
            [
                f"{page.get_text('text', clip=bb['mapped_box']).strip()}",
            ]
        )
        print("-" * 5)
    return


@app.cell
def _(first_box, page):
    page.get_text("text", clip=first_box["mapped_box"])
    return


@app.cell
def _(first_box, page, pymupdf):
    def extract_text():
        for block in page.get_textpage().extractBLOCKS():
            x, y, w, h, tt, _, _ = block
            rect = pymupdf.Rect(x, y, w, h)
            if first_box["mapped_box"].intersects(rect):
                print(f"{tt} matches")


    extract_text()
    return


@app.cell
def _(page):
    txt = "April 2025"
    rl = page.search_for(txt)
    rl
    return


@app.function
def make_text(words):
    """Return textstring output of get_text("words").

    Word items are sorted for reading sequence left to right,
    top to bottom.
    """
    line_dict = {}  # key: vertical coordinate, value: list of words
    words.sort(key=lambda w: w[0])  # sort by horizontal coordinate
    for w in words:  # fill the line dictionary
        y1 = round(w[3], 1)  # bottom of a word: don't be too picky!
        word = w[4]  # the text of the word
        line = line_dict.get(y1, [])  # read current line content
        line.append(word)  # append new word
        line_dict[y1] = line  # write back to dict
    lines = list(line_dict.items())
    lines.sort()  # sort vertically
    return "\n".join([" ".join(line[1]) for line in lines])


@app.cell
def _(
    bounding_boxes,
    fitz,
    image_h,
    image_w,
    map_image_bbox_to_pdf,
    pdf_height,
    pdf_width,
    words,
):
    for x1, y1, w1, h1 in bounding_boxes:  # From OpenCV
        pdf_rect = map_image_bbox_to_pdf(
            x1, y1, w1, h1, image_w, image_h, pdf_width, pdf_height
        )
        maybe_words = [w for w in words if fitz.Rect(w[:4]) in pdf_rect]
        text = make_text(maybe_words)

        # text = page.get_text("text", clip=pdf_rect)
        print(f"Text in region ({pdf_rect}):\n{text}")
    return


@app.cell
def _(page):
    blocks = page.get_text("dict", flags=11)["blocks"]
    blocks
    return


@app.cell
def _(bounding_boxes, fitz, page):
    bounding1 = bounding_boxes[0]
    print(bounding1)
    words = page.get_text("words")
    br = fitz.Rect(
        x0=bounding1[0],
        y0=bounding1[1],
        x1=bounding1[0] + bounding1[2],
        y1=bounding1[1] - bounding1[3],
    )
    print(br)
    return br, words


@app.cell
def _(fitz, words):
    mywords = [
        w
        for w in words
        if fitz.Rect(w[:4]).intersects(fitz.Rect(x0=70, y0=29, x1=50, y1=530))
    ]
    # mywords = [w for w in words if fitz.Rect(w[:4]).intersects(fitz.Rect(x0=65, y0=806, x1=524, y1=790))]
    mywords
    return


@app.cell
def _(br, fitz, words):
    for word in words:
        r = fitz.Rect(word[:4])

        contains = br.contains(r)
        in_ = r in br
        # print(f"{r} in {br} => {in_}", end=" ")
        # print(f"/{br} contains {r} => {contains}")
    return


if __name__ == "__main__":
    app.run()
