"""
cleanup_folder.py
Moves unwanted files and folders to _TRASH/ for review.
Nothing is permanently deleted.
"""
import os, sys, shutil
sys.stdout.reconfigure(encoding='utf-8')

ROOT  = r"d:\RESEARCH PNGS"
TRASH = os.path.join(ROOT, "_TRASH")
os.makedirs(TRASH, exist_ok=True)

moved = []
errors = []

def move(name):
    src = os.path.join(ROOT, name)
    dst = os.path.join(TRASH, name)
    if not os.path.exists(src):
        print(f"  [SKIP – not found] {name}")
        return
    try:
        shutil.move(src, dst)
        moved.append(name)
        print(f"  [MOVED] {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  [ERROR] {name}: {e}")

# ── Old / draft thesis .docx files ──────────────────────────────────────────
print("\n>>> Old thesis drafts:")
for f in [
    "123thesis.docx",
    "Chapter2_Final.docx",
    "Chapter2_Short.docx",
    "Chapter4_Final.docx",
    "Methodology_Chapter.docx",
    "Results_Chapter.docx",
    "Thesis_Chapter1_Chapter2.docx",
    "Thesis_Skin_Cancer_FYP.docx",
    "Thesis_Skin_Cancer_FYP_Formatted.docx",
    "Thesis_Skin_Cancer_FYP_v4-1.docx",
    "Thesis_Skin_Cancer_FYP_v4-1_SK1.docx",
]:
    move(f)

# ── One-off / scratch Python scripts ────────────────────────────────────────
print("\n>>> One-off Python scripts:")
for f in [
    "add_clickable_tocs.py",
    "add_plain_language.py",
    "build_architectures.py",
    "build_ch4.py",
    "build_thesis_sk1.py",
    "convert_md.py",
    "draw_matplotlib.py",
    "find_history.py",
    "fix_ch2_order.py",
    "fix_thesis_v4.py",
    "format_thesis.py",
    "generate_stats_locally.py",
    "inject_stats_cells.py",
    "inject_viz_cells.py",
    "insert_architectures.py",
    "insert_diagram.py",
    "insert_tables_ch2.py",
    "inspect_doc.py",
    "inspect_nb.py",
    "read_steps.py",
    "replace_ch2.py",
    "scratch_analyze_thesis.py",
    "scratch_count_data.py",
    "search_transcript.py",
    "transplant_ch2_full.py",
    "transplant_ch4_full.py",
]:
    move(f)

# ── Temp / scratch text files ────────────────────────────────────────────────
print("\n>>> Temp / scratch text files:")
for f in [
    "ch4_structure.txt",
    "new_structure.txt",
    "scratch_ch1ch2_text.txt",
    "scratch_ch2short.txt",
    "scratch_thesis_analysis.txt",
    "temp_ch3ch4.txt",
    "temp_structure.txt",
    "temp_text.txt",
    "temp_text2.txt",
    "thesis_stage1_section.txt",
    "thesis_structure.txt",
]:
    move(f)

# ── Large unwanted image ─────────────────────────────────────────────────────
print("\n>>> Unwanted images:")
move("Gemini_Generated_Image_ka9zrrka9zrrka9z (1).png")

# ── Dead directories ─────────────────────────────────────────────────────────
print("\n>>> Dead directories:")
for d in [
    "_deprecated_wrong_dataset",
    "_old",
    "New folder",
    "temp_docx",
    "archive",
]:
    move(d)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"✅ MOVED:  {len(moved)} items  →  _TRASH/")
if errors:
    print(f"❌ ERRORS: {len(errors)}")
    for name, err in errors:
        print(f"   {name}: {err}")
print(f"{'='*55}")
print("Review _TRASH/ then delete it manually when satisfied.")
