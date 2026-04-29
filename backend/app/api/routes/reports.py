"""
Report routes — Attendance report, CSV export, PDF export, and statistics.
"""

import io
import os
from datetime import datetime
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, Response
from typing import Optional
from jinja2 import Environment, FileSystemLoader
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
