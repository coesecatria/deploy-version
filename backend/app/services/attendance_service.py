import time
from datetime import datetime
from app.core.database import get_db
from app.core.config import settings
from app.core.constants import IST

# Shared list for live notifications (consumed by /api/attendance/recent-marked)
from app.api.routes.attendance import recent_marks

async def mark_attendance(roll_no: str, source: str = "API") -> dict:
    """
    Unified attendance marking logic for both API (Photo Upload) and Real-time Stream.
    Handles login/logout detection, cooldowns, and IST timezone.
    """
    db = get_db()
    
    # 1. Lookup Student
    student = await db.students.find_one({"roll_no": roll_no}, {"_id": 0})
    if not student:
        return {"success": False, "status": "Error", "message": f"Student {roll_no} not found in database"}

    # 2. Setup Time Context
    now = datetime.now(IST)
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # 3. Check for existing record today
    existing = await db.attendance.find_one({"roll_no": roll_no, "date": today_date})

    # 4. Fetch Dynamic Settings
    config_doc = await db.settings.find_one({"_id": "global_config"})
    sys_login = config_doc["login_time"] if config_doc and "login_time" in config_doc else getattr(settings, "LOGIN_TIME", "09:30:00")
    sys_logout = config_doc["logout_time"] if config_doc and "logout_time" in config_doc else getattr(settings, "LOGOUT_TIME", "16:30:00")

    if existing:
        # Minimum Cooldown Check (2 hours) to prevent bounce effect
        login_time_str = existing.get("login_time")
        if login_time_str:
            login_time_obj = datetime.strptime(today_date + " " + login_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            if (now - login_time_obj).total_seconds() < 2 * 3600:
                return {
                    "success": True, "roll_no": roll_no, "name": student["name"],
                    "status": "Already Marked", "message": f"{student['name']} already checked in (Cooldown Active)"
                }

        # Handle LOGOUT
        logout_thresh = datetime.strptime(today_date + " " + sys_logout, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
        status = "Logged Out" if now >= logout_thresh else "Early Logout"
        
        await db.attendance.update_one(
            {"_id": existing["_id"]},
            {"$set": {"logout_time": current_time, "logout_status": status}}
        )
        msg = f"{student['name']} {status} at {current_time}"
    else:
        # Handle LOGIN
        login_thresh = datetime.strptime(today_date + " " + sys_login, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
        status = "On Time" if now <= login_thresh else "Late"

        await db.attendance.insert_one({
            "roll_no": roll_no,
            "name": student["name"],
            "branch": student["branch"],
            "date": today_date,
            "login_time": current_time,
            "login_status": status,
            "logout_time": None,
            "logout_status": None,
        })
        msg = f"{student['name']} Logged In ({status})"

    # 5. Push to Live Notifications
    recent_marks.append({
        "roll_no": roll_no,
        "name": student["name"],
        "message": f"[{source}] {msg}",
        "timestamp": time.time()
    })
    if len(recent_marks) > 20: recent_marks.pop(0)

    return {
        "success": True, "roll_no": roll_no, "name": student["name"],
        "branch": student["branch"], "status": status, "message": msg
    }
