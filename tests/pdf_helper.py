"""A minimal valid PDF with one line of Helvetica text per page (mirrors tests/helpers/pdf.ts)."""


def make_pdf(pages):
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids = [4 + i * 2 for i in range(len(pages))]
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>"
    for page_id, text in zip(page_ids, pages):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        )
        objects[page_id + 1] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"

    pdf = "%PDF-1.4\n"
    offsets = {}
    size = max(objects) + 1
    for obj_id in range(1, size):
        offsets[obj_id] = len(pdf)
        pdf += f"{obj_id} 0 obj\n{objects[obj_id]}\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {size}\n0000000000 65535 f \n"
    pdf += "".join(f"{offsets[i]:010d} 00000 n \n" for i in range(1, size))
    pdf += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return pdf.encode("latin-1")
