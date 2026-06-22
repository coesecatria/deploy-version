"""
Report routes — Attendance report, CSV export, PDF export, and statistics.
"""

import io
import os
import base64
from collections import defaultdict
from datetime import datetime
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, Response
from typing import Optional
from jinja2 import Environment, FileSystemLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.core.database import get_db
from app.core.constants import IST
from app.models.schemas import AttendanceRecord

router = APIRouter(tags=["Reports"])

# Resolve template directory relative to this file
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
_jinja_env = Environment(loader=FileSystemLoader(os.path.abspath(_TEMPLATES_DIR)))


@router.get("/attendance-report", response_model=list[AttendanceRecord])
async def get_attendance_report(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    roll_no: Optional[str] = Query(None, description="Filter by roll number"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get attendance records with optional filters."""
    db = get_db()
    query = {}

    if date:
        query["date"] = date
    if roll_no:
        query["roll_no"] = roll_no.upper()
    if branch:
        query["branch"] = branch.upper()

    cursor = (
        db.attendance.find(query, {"_id": 0})
        .sort([("date", -1), ("login_time", -1)])
        .skip(skip)
        .limit(limit)
    )
    records = await cursor.to_list(length=limit)
    return records


@router.get("/attendance-report/csv")
async def export_attendance_csv(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
):
    """Export attendance report as CSV file."""
    db = get_db()
    query = {}

    if date:
        query["date"] = date
    if branch:
        query["branch"] = branch.upper()

    cursor = db.attendance.find(query, {"_id": 0}).sort([("date", -1), ("login_time", -1)])
    records = await cursor.to_list(length=10000)

    # Build CSV
    lines = ["Roll No,Name,Branch,Date,Login Time,Login Status,Logout Time,Logout Status"]
    for r in records:
        lines.append(f"{r['roll_no']},{r['name']},{r['branch']},{r['date']},{r.get('login_time', '')},{r.get('login_status', '')},{r.get('logout_time', '')},{r.get('logout_status', '')}")

    csv_content = "\n".join(lines)
    filename = f"attendance_report_{date or 'all'}.csv"

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/attendance-report/pdf")
async def export_attendance_pdf(
    date: Optional[str] = Query(None, description="Filter by specific date (YYYY-MM-DD). Omit for all records."),
    branch: Optional[str] = Query(None, description="Filter by branch"),
):
    """Export attendance report as a formatted PDF using Jinja2 + xhtml2pdf."""
    from xhtml2pdf import pisa

    db = get_db()
    query = {}
    if date:
        query["date"] = date
    if branch:
        query["branch"] = branch.upper()

    cursor = db.attendance.find(query, {"_id": 0}).sort([("date", -1), ("login_time", 1)])
    records = await cursor.to_list(length=10000)

    # ── Compute summary stats ──
    total = len(records)
    on_time      = sum(1 for r in records if r.get("login_status") == "On Time")
    late         = sum(1 for r in records if r.get("login_status") == "Late")
    logged_out   = sum(1 for r in records if r.get("logout_status") == "Logged Out")
    early_logout = sum(1 for r in records if r.get("logout_status") == "Early Logout")

    # ── Context for template ──
    generated_at  = datetime.now(IST).strftime("%d %b %Y, %I:%M %p")
    period_label  = date if date else "All Records"
    branch_filter = branch.upper() if branch else None

    # ── Compute Charts ──
    branch_counts = defaultdict(int)
    month_counts = defaultdict(int)

    for r in records:
        b = r.get("branch", "Unknown")
        d = r.get("date", "")
        branch_counts[b] += 1
        if d and len(d) >= 7:
            month = d[:7]  # YYYY-MM
            month_counts[month] += 1

    bar_chart_base64 = ""
    doughnut_chart_base64 = ""

    if records:
        # Generate Doughnut Chart
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        labels = list(branch_counts.keys())
        sizes = list(branch_counts.values())
        if sizes:
            # Adjust font sizes and distance to avoid overlap
            wedges, texts, autotexts = ax1.pie(
                sizes, 
                labels=labels, 
                autopct='%1.1f%%', 
                startangle=90, 
                wedgeprops=dict(width=0.3, edgecolor='w'),
                pctdistance=0.8,
                labeldistance=1.15,
                textprops={'fontsize': 8}
            )
            for autotext in autotexts:
                autotext.set_fontsize(7)
                autotext.set_color('white')
            
            ax1.axis('equal')
            plt.title('Branch-wise Attendance', fontsize=10, pad=10)
            plt.tight_layout()
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', transparent=True, dpi=150)
            buf1.seek(0)
            doughnut_chart_base64 = base64.b64encode(buf1.read()).decode('utf-8')
            plt.close(fig1)

        # Generate Bar Chart
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        m_labels = sorted(list(month_counts.keys()))
        m_sizes = [month_counts[m] for m in m_labels]
        if m_sizes:
            bars = ax2.bar(m_labels, m_sizes, color='#0d9488', width=0.6)
            ax2.set_ylabel('Attendance Count', fontsize=8)
            ax2.set_title('Monthly Attendance', fontsize=10, pad=10)
            ax2.tick_params(axis='both', which='major', labelsize=8)
            
            # Remove top and right borders for a cleaner look
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            plt.xticks(rotation=0)  # Straight text is usually better if few months
            plt.tight_layout()
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', transparent=True, dpi=150)
            buf2.seek(0)
            bar_chart_base64 = base64.b64encode(buf2.read()).decode('utf-8')
            plt.close(fig2)

    # ── Group Records for Summary Report (if no date specified) ──
    is_summary = date is None
    summary_data = {}
    
    if is_summary:
        for r in records:
            b = r.get("branch", "Unknown")
            roll = r.get("roll_no")
            name = r.get("name")
            if b not in summary_data:
                summary_data[b] = {}
            if roll not in summary_data[b]:
                summary_data[b][roll] = {
                    "name": name,
                    "total_present": 0,
                    "on_time": 0,
                    "late": 0
                }
            
            summary_data[b][roll]["total_present"] += 1
            if r.get("login_status") == "On Time":
                summary_data[b][roll]["on_time"] += 1
            elif r.get("login_status") == "Late":
                summary_data[b][roll]["late"] += 1
                
        # Convert to list and sort by roll number
        for b in summary_data:
            summary_data[b] = sorted(
                [{"roll_no": k, **v} for k, v in summary_data[b].items()],
                key=lambda x: x["roll_no"]
            )
        # Sort branches alphabetically
        summary_data = dict(sorted(summary_data.items()))

    template = _jinja_env.get_template("attendance_report.html")
    html_content = template.render(
        records=records,
        generated_at=generated_at,
        period_label=period_label,
        branch_filter=branch_filter,
        total=total,
        on_time=on_time,
        late=late,
        logged_out=logged_out,
        early_logout=early_logout,
        bar_chart_base64=bar_chart_base64,
        doughnut_chart_base64=doughnut_chart_base64,
        is_summary=is_summary,
        summary_data=summary_data,
    )

    # ── Render PDF with xhtml2pdf ──
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(src=html_content, dest=pdf_buffer)
    if pisa_status.err:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    pdf_bytes = pdf_buffer.getvalue()
    filename  = f"attendance_report_{date or 'all'}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/attendance-stats")
async def get_attendance_stats(
    date: Optional[str] = Query(None, description="Date for stats (YYYY-MM-DD), defaults to today"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
):
    """Get attendance statistics — total present, percentage, branch-wise breakdown."""
    db = get_db()

    if not date:
        date = datetime.now(IST).strftime("%Y-%m-%d")

    # Total students
    student_query = {}
    if branch:
        student_query["branch"] = branch.upper()
    total_students = await db.students.count_documents(student_query)

    # Present today
    attendance_query = {"date": date}
    if branch:
        attendance_query["branch"] = branch.upper()
    present_count = await db.attendance.count_documents(attendance_query)

    # Branch-wise breakdown
    pipeline = [
        {"$match": {"date": date}},
        {"$group": {"_id": "$branch", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    branch_stats = {}
    async for doc in db.attendance.aggregate(pipeline):
        branch_stats[doc["_id"]] = doc["count"]

    # Per-branch totals
    branch_totals_pipeline = [
        {"$group": {"_id": "$branch", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    branch_totals = {}
    async for doc in db.students.aggregate(branch_totals_pipeline):
        branch_totals[doc["_id"]] = doc["count"]

    # Compute percentages
    branch_breakdown = {}
    for b in branch_totals:
        present = branch_stats.get(b, 0)
        total = branch_totals[b]
        branch_breakdown[b] = {
            "present": present,
            "total": total,
            "percentage": round(present / total * 100, 1) if total > 0 else 0,
        }

    return {
        "date": date,
        "total_students": total_students,
        "present": present_count,
        "absent": total_students - present_count,
        "percentage": round(present_count / total_students * 100, 1) if total_students > 0 else 0,
        "branch_breakdown": branch_breakdown,
    }
