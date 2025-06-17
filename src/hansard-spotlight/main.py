import cv2
import pymupdf
import numpy as np
from PIL import Image


examples = [
    'dataset/national_assembly/Hansard Report - Wednesday, 30th April 2025 (A).pdf',
    'dataset/national_assembly/Hansard Report - Wednesday, 30th April 2025 (P).pdf',
    'dataset/national_assembly/Hansard Report - Wednesday, 23rd April 2025 (A).pdf',
]

def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
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

def main():
    doc = pymupdf.open(examples[0])
    page = doc[1]
    pix = page.get_pixmap()

    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    cv_img = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2GRAY)
    cv_img.imsho


if __name__ == '__main__':
    main()
