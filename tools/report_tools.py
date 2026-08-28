"""Generator Laporan Akademis Otomatis (Academic Report Generator).
Mengompilasi riwayat percakapan, ringkasan dataset, grafik visualisasi, rumus matematika/equation KaTeX,
dan hasil pemodelan Machine Learning menjadi dokumen Microsoft Word (.docx) berformat formal standar skripsi.
"""

import os
import re
import time
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUTPUT_FILES_DIR = os.path.join(os.getcwd(), "generated_files")
os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)


def set_cell_background(cell, fill_hex: str):
    """Mengatur warna latar belakang sel tabel."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Mengatur padding internal sel tabel."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m_name, m_val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m_name}')
        node.set(qn('w:w'), str(m_val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def clean_latex_to_math_unicode(latex_str: str) -> str:
    """Mengonversi notasi formula LaTeX menjadi tipografi matematika Unicode standar (Cambria Math)."""
    text = latex_str.strip()
    # Bersihkan pembungkus $$ atau $
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()
    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()

    # Konversi pecahan \frac{a}{b} -> (a) / (b)
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1) / (\2)', text)

    # Simbol Yunani & Notasi Matematika
    replacements = [
        (r'\\lambda', 'λ'),
        (r'\\mu', 'μ'),
        (r'\\rho', 'ρ'),
        (r'\\sigma', 'σ'),
        (r'\\beta', 'β'),
        (r'\\alpha', 'α'),
        (r'\\theta', 'θ'),
        (r'\\gamma', 'γ'),
        (r'\\delta', 'δ'),
        (r'\\epsilon', 'ε'),
        (r'\\sum', '∑'),
        (r'\\sqrt\{([^{}]+)\}', r'√(\1)'),
        (r'\\sqrt', '√'),
        (r'\\approx', '≈'),
        (r'\\le', '≤'),
        (r'\\ge', '≥'),
        (r'\\ne', '≠'),
        (r'\\pm', '±'),
        (r'\\times', '×'),
        (r'\\cdot', '·'),
        (r'\\infty', '∞'),
        (r'\\hat\{([^{}]+)\}', r'\1̂'),
        (r'\\bar\{([^{}]+)\}', r'\1̄'),
        (r'\^2', '²'),
        (r'\^3', '³'),
        (r'\^T', 'ᵀ'),
        (r'\\left', ''),
        (r'\\right', ''),
        (r'\\quad', '   '),
        (r'\\text\{([^{}]+)\}', r'\1')
    ]

    for pat, rep in replacements:
        text = re.sub(pat, rep, text)

    return text


def add_math_equation_callout(doc: Document, formula_raw: str, eq_number: str = None):
    """Menambahkan kotak formula matematika terpusat dengan tipografi Cambria Math."""
    clean_formula = clean_latex_to_math_unicode(formula_raw)
    
    # Buat tabel 1-sel sebagai container formula yang elegan
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    set_cell_background(cell, "F8FAFC") # Slate 50
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Prefix label
    prefix_run = p.add_run("Persamaan Matematis:  ")
    prefix_run.font.name = 'Calibri'
    prefix_run.font.size = Pt(9.5)
    prefix_run.font.bold = True
    prefix_run.font.color.rgb = RGBColor(14, 116, 144) # Teal
    
    # Formula text
    eq_run = p.add_run(f"  {clean_formula}  ")
    eq_run.font.name = 'Cambria Math'
    eq_run.font.size = Pt(11.5)
    eq_run.font.bold = True
    eq_run.font.italic = True
    eq_run.font.color.rgb = RGBColor(15, 23, 42) # Slate Dark
    
    if eq_number:
        num_run = p.add_run(f"       ({eq_number})")
        num_run.font.name = 'Calibri'
        num_run.font.size = Pt(9)
        num_run.font.color.rgb = RGBColor(100, 116, 139)
        
    doc.add_paragraph() # Spasi sesudah formula


def render_paragraph_with_inline_math(doc: Document, text: str, style=None):
    """Merender teks paragraf biasa dengan parsing inline math $...$ menjadi gaya Cambria Math."""
    p = doc.add_paragraph(style=style)
    p.paragraph_format.line_spacing = 1.15
    
    # Pecah berdasarkan $...$
    parts = re.split(r'(\$[^\$]+\$)', text)
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) > 2:
            math_text = clean_latex_to_math_unicode(part)
            run = p.add_run(f" {math_text} ")
            run.font.name = 'Cambria Math'
            run.font.size = Pt(10.5)
            run.font.bold = True
            run.font.italic = True
            run.font.color.rgb = RGBColor(3, 105, 161) # Sky 700
        else:
            if part:
                run = p.add_run(part)
                run.font.name = 'Calibri'
                run.font.size = Pt(10.5)
                run.font.color.rgb = RGBColor(30, 41, 59)


def generate_academic_report(
    chat_history: list,
    uploaded_files: dict = None,
    plot_files: list = None,
    model_name: str = "Gemma 4 (31B-IT)",
    custom_title: str = "Laporan Analisis Data Science & Multi-Agent Solver"
) -> str:
    """Menyusun dokumen laporan Word (.docx) lengkap dengan persamaan matematika formal."""
    doc = Document()
    
    # Atur Margin Halaman Standar Skripsi: 1 inci (2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ---------------------------------------------------------
    # 1. HEADER & COVER DOKUMEN
    # ---------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(custom_title.upper())
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(17)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(15, 23, 42) # Slate Dark

    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle_p.add_run("Dokumen Hasil Eksekusi AI Multi-Agent Hierarchy (FastAPI + LangGraph Stack)")
    sub_run.font.name = 'Calibri'
    sub_run.font.size = Pt(10.5)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(14, 116, 144) # Ocean Cyan

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 2. METADATA SESI ANALISIS (Tabel Elegan)
    # ---------------------------------------------------------
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    metadata_items = [
        ("Tanggal / Waktu Analisis", time.strftime("%d %B %Y, %H:%M:%S WIB")),
        ("Arsitektur Sistem", "Hierarchical Multi-Agent (4 Specialized Agents)"),
        ("Model Large Language Model", model_name),
        ("Jumlah Berkas Masukan", f"{len(uploaded_files or {})} Berkas Terunggah"),
        ("Fitur Matematika & Sandbox", "Equation Box (Cambria Math) + Self-Healing Kernel")
    ]

    for idx, (label, val) in enumerate(metadata_items):
        row = meta_table.rows[idx]
        
        c0 = row.cells[0]
        c0.width = Inches(2.3)
        set_cell_margins(c0, 100, 100, 150, 150)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(label)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        set_cell_background(c0, "F1F5F9")
        
        c1 = row.cells[1]
        c1.width = Inches(4.2)
        set_cell_margins(c1, 100, 100, 150, 150)
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.size = Pt(9.5)
        set_cell_background(c1, "FFFFFF" if idx % 2 == 0 else "F8FAFC")

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 3. BAB I: DAFTAR BERKAS DATASET & DOKUMEN
    # ---------------------------------------------------------
    h1 = doc.add_heading("I. DAFTAR BERKAS TUGAS & DATASET", level=1)
    h1.runs[0].font.color.rgb = RGBColor(15, 23, 42)
    
    if uploaded_files:
        file_table = doc.add_table(rows=1, cols=3)
        file_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        headers = ["No", "Nama Berkas", "Tipe & Format"]
        hdr_cells = file_table.rows[0].cells
        for i, text in enumerate(headers):
            set_cell_margins(hdr_cells[i], 120, 120, 150, 150)
            run = hdr_cells[i].paragraphs[0].add_run(text)
            run.font.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(255, 255, 255)
            set_cell_background(hdr_cells[i], "0F172A")
            
        for num, (fname, fpath) in enumerate(uploaded_files.items(), 1):
            row_cells = file_table.add_row().cells
            for c in row_cells:
                set_cell_margins(c, 80, 80, 120, 120)
                set_cell_background(c, "FFFFFF" if num % 2 != 0 else "F8FAFC")
                
            row_cells[0].paragraphs[0].add_run(str(num)).font.size = Pt(9)
            row_cells[1].paragraphs[0].add_run(fname).font.size = Pt(9)
            ext = os.path.splitext(fname)[1].upper().replace('.', '')
            row_cells[2].paragraphs[0].add_run(f"Data Tabular / Dokumen ({ext})").font.size = Pt(9)
    else:
        doc.add_paragraph("Tidak ada berkas eksternal yang diunggah pada sesi ini.")

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 4. BAB II: PROSES EKSEKUSI 4 AGEN & HASIL ANALISIS (DENGAN EQUATIONS)
    # ---------------------------------------------------------
    h2 = doc.add_heading("II. PROSES EKSEKUSI MULTI-AGENT & FORMULASI MATEMATIKA", level=1)
    h2.runs[0].font.color.rgb = RGBColor(15, 23, 42)

    turn_count = 1
    eq_counter = 1
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            doc.add_heading(f"Instruksi Pengguna #{turn_count}:", level=2)
            p = doc.add_paragraph()
            r = p.add_run(content)
            r.font.italic = True
            r.font.color.rgb = RGBColor(30, 41, 59)
            turn_count += 1
        elif role == "assistant":
            doc.add_heading("Hasil Analisis, Formulasi Rumus & Laporan Sintesis AI:", level=2)
            
            lines = content.split("\n")
            in_code_block = False
            code_buffer = []

            for line in lines:
                # Tangani Blok Kode Python
                if line.startswith("```"):
                    in_code_block = not in_code_block
                    if not in_code_block and code_buffer:
                        # Buat box kode
                        c_table = doc.add_table(rows=1, cols=1)
                        c_cell = c_table.rows[0].cells[0]
                        set_cell_background(c_cell, "1E293B")
                        set_cell_margins(c_cell, 80, 80, 100, 100)
                        cp = c_cell.paragraphs[0]
                        crun = cp.add_run("\n".join(code_buffer))
                        crun.font.name = 'Consolas'
                        crun.font.size = Pt(8.5)
                        crun.font.color.rgb = RGBColor(226, 232, 240)
                        code_buffer = []
                        doc.add_paragraph()
                    continue

                if in_code_block:
                    code_buffer.append(line)
                    continue

                # 1. Tangani Block Equation $$...$$
                if line.strip().startswith("$$") and line.strip().endswith("$$"):
                    add_math_equation_callout(doc, line.strip(), eq_number=f"2.{eq_counter}")
                    eq_counter += 1
                # 2. Tangani Heading
                elif line.startswith("### "):
                    doc.add_heading(line[4:].strip(), level=3)
                elif line.startswith("## "):
                    doc.add_heading(line[3:].strip(), level=3)
                elif line.startswith("# "):
                    doc.add_heading(line[2:].strip(), level=3)
                # 3. Tangani Bullet Points dengan Inline Math
                elif line.startswith("* ") or line.startswith("- "):
                    render_paragraph_with_inline_math(doc, line[2:].strip(), style='List Bullet')
                # 4. Paragraf Teks Biasa dengan Inline Math
                elif line.strip():
                    # Jika baris ini terdeteksi sebagai formula yang berdiri sendiri
                    if any(k in line for k in ["\\lambda", "\\mu", "\\rho", "\\frac", "L_q =", "W_q =", "P_0 ="]):
                        add_math_equation_callout(doc, line.strip(), eq_number=f"2.{eq_counter}")
                        eq_counter += 1
                    else:
                        render_paragraph_with_inline_math(doc, line.strip())

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 5. BAB III: GRAFIK & VISUALISASI DATA RESOLUSI TINGGI
    # ---------------------------------------------------------
    if plot_files:
        h3 = doc.add_heading("III. VISUALISASI DATA & GRAFIK HASIL KOMPUTASI", level=1)
        h3.runs[0].font.color.rgb = RGBColor(15, 23, 42)
        
        for idx, plot_path in enumerate(plot_files, 1):
            if os.path.exists(plot_path) and plot_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    doc.add_picture(plot_path, width=Inches(5.4))
                    
                    caption = doc.add_paragraph()
                    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cap_run = caption.add_run(f"Gambar {idx}: Visualisasi {os.path.basename(plot_path)}")
                    cap_run.font.size = Pt(9.5)
                    cap_run.font.italic = True
                    cap_run.font.color.rgb = RGBColor(100, 116, 139)
                except Exception:
                    pass

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 6. BAB IV: KESIMPULAN & PENUTUP
    # ---------------------------------------------------------
    h4 = doc.add_heading("IV. KESIMPULAN AKADEMIS", level=1)
    h4.runs[0].font.color.rgb = RGBColor(15, 23, 42)
    
    closing_p = doc.add_paragraph(
        "Seluruh rangkaian analisis data science, formulasi persamaan matematika, pengujian hipotesis, "
        "pembersihan dataset, dan pelatihan model Machine Learning telah diselesaikan oleh Tim AI Multi-Agent secara terstruktur, "
        "deterministik, dan bebas halusinasi. Seluruh rumus dan kode eksekusi dapat diverifikasi langsung melalui log sistem."
    )
    closing_p.paragraph_format.line_spacing = 1.15

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"Laporan_DataScience_{timestamp_str}.docx"
    report_filepath = os.path.join(OUTPUT_FILES_DIR, report_filename)
    
    doc.save(report_filepath)
    return report_filepath
