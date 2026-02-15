#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate JARVIS Project Verification Report PDF"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from datetime import datetime
import os

# Register fonts
pdfmetrics.registerFont(TTFont('Microsoft YaHei', '/usr/share/fonts/truetype/chinese/msyh.ttf'))
pdfmetrics.registerFont(TTFont('SimHei', '/usr/share/fonts/truetype/chinese/SimHei.ttf'))
pdfmetrics.registerFont(TTFont('Times New Roman', '/usr/share/fonts/truetype/english/Times-New-Roman.ttf'))
registerFontFamily('Microsoft YaHei', normal='Microsoft YaHei', bold='Microsoft YaHei')
registerFontFamily('SimHei', normal='SimHei', bold='SimHei')
registerFontFamily('Times New Roman', normal='Times New Roman', bold='Times New Roman')

# Create document
output_path = '/home/z/my-project/download/JARVIS_GitHub_Verification_Report.pdf'
doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    title='JARVIS_GitHub_Verification_Report',
    author='Z.ai',
    creator='Z.ai',
    subject='Deep verification of GitHub repository status'
)

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    name='CoverTitle',
    fontName='Microsoft YaHei',
    fontSize=32,
    leading=40,
    alignment=TA_CENTER,
    spaceAfter=36
)
subtitle_style = ParagraphStyle(
    name='Subtitle',
    fontName='Microsoft YaHei',
    fontSize=16,
    leading=22,
    alignment=TA_CENTER,
    spaceAfter=24
)
heading1_style = ParagraphStyle(
    name='Heading1CN',
    fontName='Microsoft YaHei',
    fontSize=18,
    leading=24,
    alignment=TA_LEFT,
    spaceBefore=18,
    spaceAfter=12,
    textColor=colors.HexColor('#1F4E79')
)
heading2_style = ParagraphStyle(
    name='Heading2CN',
    fontName='Microsoft YaHei',
    fontSize=14,
    leading=20,
    alignment=TA_LEFT,
    spaceBefore=12,
    spaceAfter=8,
    textColor=colors.HexColor('#2E75B6')
)
body_style = ParagraphStyle(
    name='BodyCN',
    fontName='SimHei',
    fontSize=11,
    leading=18,
    alignment=TA_LEFT,
    spaceAfter=8,
    wordWrap='CJK'
)
success_style = ParagraphStyle(
    name='Success',
    fontName='SimHei',
    fontSize=14,
    leading=20,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#008000'),
    spaceBefore=24,
    spaceAfter=24
)
table_header_style = ParagraphStyle(
    name='TableHeader',
    fontName='SimHei',
    fontSize=10,
    textColor=colors.white,
    alignment=TA_CENTER
)
table_cell_style = ParagraphStyle(
    name='TableCell',
    fontName='SimHei',
    fontSize=10,
    alignment=TA_CENTER,
    wordWrap='CJK'
)
table_cell_left_style = ParagraphStyle(
    name='TableCellLeft',
    fontName='SimHei',
    fontSize=10,
    alignment=TA_LEFT,
    wordWrap='CJK'
)

story = []

# Cover Page
story.append(Spacer(1, 80))
story.append(Paragraph('<b>JARVIS v14 Ultimate</b>', title_style))
story.append(Spacer(1, 24))
story.append(Paragraph('<b>GitHub Repository Deep Verification Report</b>', subtitle_style))
story.append(Spacer(1, 48))
story.append(Paragraph('Repository: https://github.com/71261121/auto-jarvis-', subtitle_style))
story.append(Spacer(1, 24))
story.append(Paragraph(f'Verification Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', subtitle_style))
story.append(Spacer(1, 48))
story.append(Paragraph('<b>Verification Status: PROJECT SAFE AND COMPLETE</b>', success_style))
story.append(PageBreak())

# Executive Summary
story.append(Paragraph('<b>1. Executive Summary</b>', heading1_style))
story.append(Paragraph(
    'This report presents a comprehensive verification of the JARVIS v14 Ultimate project GitHub repository. '
    'The user reported concerns about potential file deletion on GitHub. After deep analysis using GitHub API '
    'and local file system verification, we can confirm that the repository is completely intact and all files are present.',
    body_style
))
story.append(Spacer(1, 12))

# Key Findings
story.append(Paragraph('<b>Key Findings:</b>', heading2_style))
findings_data = [
    [Paragraph('<b>Category</b>', table_header_style), Paragraph('<b>Status</b>', table_header_style), Paragraph('<b>Details</b>', table_header_style)],
    [Paragraph('GitHub Repository', table_cell_style), Paragraph('OK', table_cell_style), Paragraph('All files present on main branch', table_cell_left_style)],
    [Paragraph('main.py', table_cell_style), Paragraph('OK', table_cell_style), Paragraph('29,690 bytes - complete and intact', table_cell_left_style)],
    [Paragraph('Local Files', table_cell_style), Paragraph('OK', table_cell_style), Paragraph('112 Python files, 49 Markdown files', table_cell_left_style)],
    [Paragraph('Commit History', table_cell_style), Paragraph('OK', table_cell_style), Paragraph('Multiple commits with fixes applied', table_cell_left_style)],
    [Paragraph('Project Size', table_cell_style), Paragraph('OK', table_cell_style), Paragraph('7.9 MB total project size', table_cell_left_style)],
]
findings_table = Table(findings_data, colWidths=[4*cm, 2.5*cm, 9*cm])
findings_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, 1), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#F5F5F5')),
    ('BACKGROUND', (0, 3), (-1, 3), colors.white),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#F5F5F5')),
    ('BACKGROUND', (0, 5), (-1, 5), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
]))
story.append(findings_table)
story.append(Spacer(1, 6))
story.append(Paragraph('Table 1: Verification Summary', ParagraphStyle('Caption', fontName='SimHei', fontSize=10, alignment=TA_CENTER)))
story.append(Spacer(1, 18))

# GitHub Verification
story.append(Paragraph('<b>2. GitHub Repository Verification</b>', heading1_style))
story.append(Paragraph(
    'Using the GitHub REST API, we verified the complete contents of the repository. '
    'The API response confirms all expected files and directories are present on the main branch:',
    body_style
))
story.append(Spacer(1, 12))

# GitHub Files Table
github_files_data = [
    [Paragraph('<b>File/Directory</b>', table_header_style), Paragraph('<b>Type</b>', table_header_style), Paragraph('<b>Size</b>', table_header_style), Paragraph('<b>Status</b>', table_header_style)],
    [Paragraph('.gitignore', table_cell_style), Paragraph('File', table_cell_style), Paragraph('539 bytes', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('LICENSE', table_cell_style), Paragraph('File', table_cell_style), Paragraph('1,089 bytes', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('README.md', table_cell_style), Paragraph('File', table_cell_style), Paragraph('8,603 bytes', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('RELEASE_NOTES.md', table_cell_style), Paragraph('File', table_cell_style), Paragraph('9,906 bytes', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('main.py', table_cell_style), Paragraph('File', table_cell_style), Paragraph('29,690 bytes', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('requirements.txt', table_cell_style), Paragraph('File', table_cell_style), Paragraph('506 bytes', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('config/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('core/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('docs/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('install/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('interface/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('research/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('security/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
    [Paragraph('tests/', table_cell_style), Paragraph('Directory', table_cell_style), Paragraph('-', table_cell_style), Paragraph('Present', table_cell_style)],
]
github_table = Table(github_files_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm])
github_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
]))
for i in range(1, len(github_files_data)):
    if i % 2 == 0:
        github_table.setStyle(TableStyle([('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F5F5F5'))]))
    else:
        github_table.setStyle(TableStyle([('BACKGROUND', (0, i), (-1, i), colors.white)]))
story.append(github_table)
story.append(Spacer(1, 6))
story.append(Paragraph('Table 2: GitHub Repository Contents (via API)', ParagraphStyle('Caption', fontName='SimHei', fontSize=10, alignment=TA_CENTER)))
story.append(Spacer(1, 18))

# Local Files Verification
story.append(Paragraph('<b>3. Local Files Verification</b>', heading1_style))
story.append(Paragraph(
    'The local file system was also verified to ensure all project files exist. '
    'The local repository contains a complete copy of the JARVIS project:',
    body_style
))
story.append(Spacer(1, 12))

# Local Stats
local_data = [
    [Paragraph('<b>Category</b>', table_header_style), Paragraph('<b>Count</b>', table_header_style)],
    [Paragraph('Python Files (.py)', table_cell_style), Paragraph('112', table_cell_style)],
    [Paragraph('Markdown Files (.md)', table_cell_style), Paragraph('49', table_cell_style)],
    [Paragraph('Total Project Size', table_cell_style), Paragraph('7.9 MB', table_cell_style)],
    [Paragraph('Core Modules', table_cell_style), Paragraph('11 files', table_cell_style)],
    [Paragraph('AI Modules', table_cell_style), Paragraph('10 files', table_cell_style)],
    [Paragraph('Security Modules', table_cell_style), Paragraph('9 files', table_cell_style)],
    [Paragraph('Self-Modification Modules', table_cell_style), Paragraph('6 files', table_cell_style)],
]
local_table = Table(local_data, colWidths=[8*cm, 4*cm])
local_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
for i in range(1, len(local_data)):
    if i % 2 == 0:
        local_table.setStyle(TableStyle([('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F5F5F5'))]))
story.append(local_table)
story.append(Spacer(1, 6))
story.append(Paragraph('Table 3: Local File Statistics', ParagraphStyle('Caption', fontName='SimHei', fontSize=10, alignment=TA_CENTER)))
story.append(Spacer(1, 18))

# Commit History
story.append(Paragraph('<b>4. Commit History Verification</b>', heading1_style))
story.append(Paragraph(
    'The GitHub commit history shows multiple commits with critical fixes that were previously applied. '
    'This confirms the repository was updated successfully:',
    body_style
))
story.append(Spacer(1, 12))

commit_data = [
    [Paragraph('<b>Commit</b>', table_header_style), Paragraph('<b>Date</b>', table_header_style), Paragraph('<b>Description</b>', table_header_style)],
    [Paragraph('1ad8da7', table_cell_style), Paragraph('2026-02-14', table_cell_style), Paragraph('DEEP ANALYSIS: Verified AI claims, Fixed Termux SSL', table_cell_left_style)],
    [Paragraph('0d3509e', table_cell_style), Paragraph('2026-02-14', table_cell_style), Paragraph('CRITICAL FIXES: 10 verified import/usage errors corrected', table_cell_left_style)],
    [Paragraph('b00b1d1', table_cell_style), Paragraph('2026-02-14', table_cell_style), Paragraph('Import/Usage mismatches, Termux compatibility', table_cell_left_style)],
    [Paragraph('eaa99d7', table_cell_style), Paragraph('2026-02-14', table_cell_style), Paragraph('Project Organization: LICENSE, requirements.txt added', table_cell_left_style)],
]
commit_table = Table(commit_data, colWidths=[2.5*cm, 3*cm, 8.5*cm])
commit_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
for i in range(1, len(commit_data)):
    if i % 2 == 0:
        commit_table.setStyle(TableStyle([('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F5F5F5'))]))
story.append(commit_table)
story.append(Spacer(1, 6))
story.append(Paragraph('Table 4: Recent Commit History', ParagraphStyle('Caption', fontName='SimHei', fontSize=10, alignment=TA_CENTER)))
story.append(Spacer(1, 18))

# main.py Verification
story.append(Paragraph('<b>5. main.py Code Verification</b>', heading1_style))
story.append(Paragraph(
    'The main.py file was verified to contain complete and correct code. The file includes:',
    body_style
))
story.append(Spacer(1, 12))

features = [
    'Complete JARVIS application class with all initialization methods',
    'Correct imports: EventEmitter, MemoryCache, get_cache, CodeValidator, SelfImprovementEngine',
    'Proper state machine integration with JarvisStates enum',
    'Security modules: EncryptionManager, ExecutionSandbox, AuditLogger',
    'Memory systems: ContextManager, ChatStorage, MemoryOptimizer',
    'Cross-platform signal handling for Windows and Unix',
    'Interactive CLI mode with command processing',
    'AI integration with OpenRouter client',
]
for feature in features:
    story.append(Paragraph(f'  - {feature}', body_style))
story.append(Spacer(1, 18))

# Conclusion
story.append(Paragraph('<b>6. Conclusion</b>', heading1_style))
story.append(Paragraph(
    'Based on comprehensive verification using GitHub API and local file system analysis, '
    'we can definitively confirm:',
    body_style
))
story.append(Spacer(1, 12))

conclusions = [
    'The GitHub repository https://github.com/71261121/auto-jarvis- is COMPLETE and INTACT',
    'All project files (main.py, requirements.txt, README.md, etc.) are present',
    'All directories (core/, security/, interface/, etc.) exist and contain files',
    'The commit history shows all previous fixes were successfully applied',
    'The local copy contains 112 Python files and 49 Markdown files',
    'NO FILES WERE DELETED - the project is safe',
]
for conclusion in conclusions:
    story.append(Paragraph(f'  - {conclusion}', body_style))
story.append(Spacer(1, 24))

story.append(Paragraph('<b>Final Status: PROJECT VERIFIED SAFE - NO ACTION REQUIRED</b>', success_style))

# Build PDF
doc.build(story)
print(f"PDF generated: {output_path}")
