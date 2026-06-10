from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from app.core.database import get_db
from enum import Enum

router = APIRouter(prefix="/api/fleet", tags=["fleet"])


class FleetCommandType(str, Enum):
    start_batch = "start_batch"
    stop_batch = "stop_batch"
    pause = "pause"
    resume = "resume"


class FleetScheduleCreate(BaseModel):
    required_batches: int = Field(..., ge=1)
    time_horizon_hours: int = Field(..., ge=1, le=168)
    device_ids: Optional[List[int]] = None
    priority: Optional[str] = "normal"


class FleetCommandSend(BaseModel):
    command: FleetCommandType
    batch_id: Optional[str] = None
    parameters: Optional[dict] = None


@router.post("/schedule", status_code=201)
async def create_schedule(
    schedule: FleetScheduleCreate,
    db: AsyncSession = Depends(get_db)
):
    try:
        insert_sql = text("""
            INSERT INTO fleet_schedules (
                required_batches, time_horizon_hours, device_ids, priority, status, created_at
            ) VALUES (
                :required_batches, :time_horizon_hours, :device_ids, :priority, 'pending', :created_at
            )
            RETURNING id, required_batches, time_horizon_hours, device_ids, priority, status, created_at
        """)

        values = {
            "required_batches": schedule.required_batches,
            "time_horizon_hours": schedule.time_horizon_hours,
            "device_ids": schedule.device_ids,
            "priority": schedule.priority,
            "created_at": datetime.now()
        }

        result = await db.execute(insert_sql, values)
        await db.commit()
        row = result.first()

        return {
            "id": row.id,
            "required_batches": row.required_batches,
            "time_horizon_hours": row.time_horizon_hours,
            "device_ids": row.device_ids,
            "priority": row.priority,
            "status": row.status,
            "created_at": row.created_at
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule/{schedule_id}")
async def get_schedule(
    schedule_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        query = text("""
            SELECT id, required_batches, time_horizon_hours, device_ids, priority, status, created_at, started_at, completed_at
            FROM fleet_schedules
            WHERE id = :schedule_id
        """)

        result = await db.execute(query, {"schedule_id": schedule_id})
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Schedule not found")

        return {
            "id": row.id,
            "required_batches": row.required_batches,
            "time_horizon_hours": row.time_horizon_hours,
            "device_ids": row.device_ids,
            "priority": row.priority,
            "status": row.status,
            "created_at": row.created_at,
            "started_at": row.started_at,
            "completed_at": row.completed_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules")
async def list_schedules(
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    try:
        base_query = """
            SELECT id, required_batches, time_horizon_hours, device_ids, priority, status, created_at, started_at, completed_at
            FROM fleet_schedules
        """
        conditions = []
        params = {}

        if status:
            conditions.append("status = :status")
            params["status"] = status

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY created_at DESC LIMIT :limit"
        params["limit"] = limit

        query = text(base_query)
        result = await db.execute(query, params)
        rows = result.all()

        schedules = []
        for row in rows:
            schedules.append({
                "id": row.id,
                "required_batches": row.required_batches,
                "time_horizon_hours": row.time_horizon_hours,
                "device_ids": row.device_ids,
                "priority": row.priority,
                "status": row.status,
                "created_at": row.created_at,
                "started_at": row.started_at,
                "completed_at": row.completed_at
            })

        return {"count": len(schedules), "schedules": schedules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/command/{device_id}")
async def send_fleet_command(
    device_id: int,
    command_data: FleetCommandSend,
    db: AsyncSession = Depends(get_db)
):
    try:
        insert_sql = text("""
            INSERT INTO fleet_commands (
                device_id, command, batch_id, parameters, status, created_at
            ) VALUES (
                :device_id, :command, :batch_id, :parameters, 'sent', :created_at
            )
            RETURNING id, device_id, command, batch_id, parameters, status, created_at
        """)

        values = {
            "device_id": device_id,
            "command": command_data.command.value,
            "batch_id": command_data.batch_id,
            "parameters": command_data.parameters,
            "created_at": datetime.now()
        }

        result = await db.execute(insert_sql, values)
        await db.commit()
        row = result.first()

        return {
            "id": row.id,
            "device_id": row.device_id,
            "command": row.command,
            "batch_id": row.batch_id,
            "parameters": row.parameters,
            "status": row.status,
            "created_at": row.created_at
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{device_id}")
async def get_device_fleet_status(
    device_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        query = text("""
            SELECT device_id, status, current_batch, batch_progress, current_schedule_id, last_command, last_update
            FROM fleet_status
            WHERE device_id = :device_id
            ORDER BY last_update DESC
            LIMIT 1
        """)

        result = await db.execute(query, {"device_id": device_id})
        row = result.first()

        if not row:
            return {
                "device_id": device_id,
                "status": "unknown",
                "current_batch": None,
                "batch_progress": 0,
                "current_schedule_id": None,
                "last_command": None,
                "last_update": None
            }

        return {
            "device_id": row.device_id,
            "status": row.status,
            "current_batch": row.current_batch,
            "batch_progress": row.batch_progress,
            "current_schedule_id": row.current_schedule_id,
            "last_command": row.last_command,
            "last_update": row.last_update
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overview")
async def get_fleet_overview(
    db: AsyncSession = Depends(get_db)
):
    try:
        status_query = text("""
            SELECT DISTINCT ON (device_id)
                device_id, status, current_batch, batch_progress, current_schedule_id, last_command, last_update
            FROM fleet_status
            ORDER BY device_id, last_update DESC
        """)

        result = await db.execute(status_query)
        rows = result.all()

        devices = []
        status_counts = {"idle": 0, "running": 0, "paused": 0, "error": 0, "unknown": 0}
        total_batches = 0
        avg_progress = 0.0

        for row in rows:
            device_status = row.status if row.status in status_counts else "unknown"
            status_counts[device_status] += 1
            if row.batch_progress is not None:
                total_batches += 1
                avg_progress += row.batch_progress

            devices.append({
                "device_id": row.device_id,
                "status": row.status,
                "current_batch": row.current_batch,
                "batch_progress": row.batch_progress,
                "current_schedule_id": row.current_schedule_id,
                "last_command": row.last_command,
                "last_update": row.last_update
            })

        if total_batches > 0:
            avg_progress = avg_progress / total_batches

        schedule_query = text("""
            SELECT COUNT(*) as active_count
            FROM fleet_schedules
            WHERE status IN ('pending', 'running')
        """)

        schedule_result = await db.execute(schedule_query)
        schedule_row = schedule_result.first()
        active_schedules = schedule_row.active_count if schedule_row else 0

        return {
            "total_devices": len(devices),
            "status_summary": status_counts,
            "active_schedules": active_schedules,
            "average_batch_progress": round(avg_progress, 2) if total_batches > 0 else 0,
            "devices": devices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_fleet_stats(
    hours: int = Query(24, ge=1, le=720),
    db: AsyncSession = Depends(get_db)
):
    try:
        start_time = datetime.now() - timedelta(hours=hours)

        stats_query = text("""
            SELECT
                COUNT(*) as total_commands,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_commands,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_commands,
                COUNT(DISTINCT device_id) as active_devices
            FROM fleet_commands
            WHERE created_at >= :start_time
        """)

        result = await db.execute(stats_query, {"start_time": start_time})
        row = result.first()

        total_commands = row.total_commands or 0
        completed_commands = row.completed_commands or 0
        failed_commands = row.failed_commands or 0
        active_devices = row.active_devices or 0

        success_rate = (completed_commands / total_commands * 100) if total_commands > 0 else 0.0

        schedule_stats_query = text("""
            SELECT
                COUNT(*) as total_schedules,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_schedules,
                SUM(required_batches) as total_batches_scheduled
            FROM fleet_schedules
            WHERE created_at >= :start_time
        """)

        schedule_result = await db.execute(schedule_stats_query, {"start_time": start_time})
        schedule_row = schedule_result.first()

        total_schedules = schedule_row.total_schedules or 0
        completed_schedules = schedule_row.completed_schedules or 0
        total_batches_scheduled = schedule_row.total_batches_scheduled or 0

        device_efficiency_query = text("""
            SELECT
                device_id,
                COUNT(*) as commands_executed,
                AVG(batch_progress) as avg_progress
            FROM fleet_status
            WHERE last_update >= :start_time
            GROUP BY device_id
        """)

        efficiency_result = await db.execute(device_efficiency_query, {"start_time": start_time})
        efficiency_rows = efficiency_result.all()

        device_efficiency = []
        for eff_row in efficiency_rows:
            device_efficiency.append({
                "device_id": eff_row.device_id,
                "commands_executed": eff_row.commands_executed,
                "avg_progress": round(float(eff_row.avg_progress), 2) if eff_row.avg_progress else 0
            })

        optimization_score = 0.0
        if total_schedules > 0 and active_devices > 0:
            schedule_completion_rate = (completed_schedules / total_schedules * 100) if total_schedules > 0 else 0
            optimization_score = (success_rate * 0.4 + schedule_completion_rate * 0.4 + (active_devices * 10) * 0.2)

        return {
            "time_window_hours": hours,
            "commands": {
                "total": total_commands,
                "completed": completed_commands,
                "failed": failed_commands,
                "success_rate": round(success_rate, 2)
            },
            "schedules": {
                "total": total_schedules,
                "completed": completed_schedules,
                "total_batches_scheduled": total_batches_scheduled
            },
            "active_devices": active_devices,
            "optimization_score": round(optimization_score, 2),
            "device_efficiency": device_efficiency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
