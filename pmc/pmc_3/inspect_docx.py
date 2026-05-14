import docx

doc = docx.Document('PMC3.docx')
for i, table in enumerate(doc.tables):
    print(f"Table {i}: {len(table.rows)} rows, {len(table.columns)} columns")
    if len(table.rows) > 0:
        for j, row in enumerate(table.rows[:5]):
            print(f"  Row {j}: {[cell.text.strip() for cell in row.cells]}")
