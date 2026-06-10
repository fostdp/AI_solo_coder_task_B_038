from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from app.core.database import get_db


class DefrostCommand(BaseModel):
    command: str = Field(..., pattern="^(start|stop|cancel)$")
    heating_power_pct: Optional[float] = Field(None, ge=0, le=100)
    max_duration_minutes: Optional[int] = Field(None, ge=1)


router = APIRouter(prefix="/api/defrost", tags=["defrost"])


@router.get("/optimization/{device_id}")
async def get_defrost_optimization_history(
    device_id: int,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    try:
        query = text("""
            SELECT * FROM defrost_optimizations
            WHERE device_id = :device_id
            ORDER BY timestamp DESC
            LIMIT :limit
        """)

        result = await db.execute(query, {"device_id": device_id, "limit": limit})
        rows = result.all()

        history = []
        for row in rows:
            history.append({
                "id": row.id,
                "device_id": row.device_id,
                "timestamp": row.timestamp,
                "optimization_type": row.optimization_type,
                "recommended_action": row.recommended_action,
                "estimated_frost_thickness_mm": row.estimated_frost_thickness_mm,
                "predicted_energy_saving_kwh": row.predicted_energy_saving_kwh,
                "confidence_score": row.confidence_score,
                "scheduled_time": row.scheduled_time,
                "is_approved": row.is_approved
            })

        return {"device_id": device_id, "count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current/{device_id}")
async def get_current_defrost_status(
    device_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        status_query = text("""
            SELECT * FROM defrost_status
            WHERE device_id = :device_id
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        status_result = await db.execute(status_query, {"device_id": device_id})
        status_row = status_result.first()

        telemetry_query = text("""
            SELECT cold_trap_temp, vacuum_1, vacuum_2, timestamp
            FROM telemetry
            WHERE device_id = :device_id
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        telemetry_result = await db.execute(telemetry_query, {"device_id": device_id})
        telemetry_row = telemetry_result.first()

        base_cold_trap_temp = -80.0
        calibration_factor = 1.2
        max_frost_thickness_mm = 5.0

        if telemetry_row and telemetry_row.cold_trap_temp is not None:
            temp_diff = telemetry_row.cold_trap_temp - base_cold_trap_temp
            frost_thickness_mm = max(0.0, min(max_frost_thickness_mm, temp_diff * calibration_factor * 0.1))
        else:
            frost_thickness_mm = None

        return {
            "device_id": device_id,
            "timestamp": status_row.timestamp if status_row else datetime.now(),
            "is_defrosting": status_row.is_defrosting if status_row else False,
            "defrost_phase": status_row.defrost_phase if status_row else "idle",
            "current_heating_power_pct": status_row.current_heating_power_pct if status_row else 0,
            "elapsed_minutes": status_row.elapsed_minutes if status_row else 0,
            "remaining_minutes": status_row.remaining_minutes if status_row else 0,
            "estimated_frost_thickness_mm": frost_thickness_mm,
            "cold_trap_temp": telemetry_row.cold_trap_temp if telemetry_row else None,
            "vacuum_levels": [telemetry_row.vacuum_1, telemetry_row.vacuum_2] if telemetry_row else [None, None],
            "last_telemetry_timestamp": telemetry_row.timestamp if telemetry_row else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/command/{device_id}", status_code=201)
async def send_defrost_command(
    device_id: int,
    command: DefrostCommand,
    db: AsyncSession = Depends(get_db)
):
    try:
        values = {
            "device_id": device_id,
            "command": command.command,
            "heating_power_pct": command.heating_power_pct,
            "max_duration_minutes": command.max_duration_minutes,
            "timestamp": datetime.now()
        }

        insert_sql = text("""
            INSERT INTO defrost_commands (
                device_id, command, heating_power_pct, max_duration_minutes, timestamp
            ) VALUES (
                :device_id, :command, :heating_power_pct, :max_duration_minutes, :timestamp
            )
        """)

        await db.execute(insert_sql, values)
        await db.commit()

        status_update_sql = text("""
            INSERT INTO defrost_status (
                device_id, is_defrosting, defrost_phase, current_heating_power_pct,
                elapsed_minutes, remaining_minutes, timestamp
            ) VALUES (
                :device_id, :is_defrosting, :defrost_phase, :current_heating_power_pct,
                :elapsed_minutes, :remaining_minutes, :timestamp
            )
        """)

        if command.command == "start":
            await db.execute(status_update_sql, {
                "device_id": device_id,
                "is_defrosting": True,
                "defrost_phase": "preheating",
                "current_heating_power_pct": command.heating_power_pct or 30.0,
                "elapsed_minutes": 0,
                "remaining_minutes": command.max_duration_minutes or 60,
                "timestamp": datetime.now()
            })
        elif command.command in ["stop", "cancel"]:
            await db.execute(status_update_sql, {
                "device_id": device_id,
                "is_defrosting": False,
                "defrost_phase": "idle",
                "current_heating_power_pct": 0,
                "elapsed_minutes": 0,
                "remaining_minutes": 0,
                "timestamp": datetime.now()
            })

        await db.commit()

        return {
            "status": "success",
            "message": f"Defrost {command.command} command sent",
            "device_id": device_id,
            "command": command.command,
            "heating_power_pct": command.heating_power_pct,
            "max_duration_minutes": command.max_duration_minutes
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{device_id}")
async def get_defrost_status_history(
    device_id: int,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    try:
        query = text("""
            SELECT * FROM defrost_status
            WHERE device_id = :device_id
            ORDER BY timestamp DESC
            LIMIT :limit
        """)

        result = await db.execute(query, {"device_id": device_id, "limit": limit})
        rows = result.all()

        history = []
        for row in rows:
            history.append({
                "id": row.id,
                "device_id": row.device_id,
                "timestamp": row.timestamp,
                "is_defrosting": row.is_defrosting,
                "defrost_phase": row.defrost_phase,
                "current_heating_power_pct": row.current_heating_power_pct,
                "elapsed_minutes": row.elapsed_minutes,
                "remaining_minutes": row.remaining_minutes,
                "target_temp": row.target_temp,
                "current_temp": row.current_temp
            })

        return {"device_id": device_id, "count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{device_id}")
async def get_defrost_energy_saving_stats(
    device_id: int,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    try:
        start_time = datetime.now() - timedelta(days=days)

        stats_query = text("""
            SELECT
                COUNT(*) as total_defrost_count,
                SUM(CASE WHEN is_approved THEN 1 ELSE 0 END) as approved_count,
                SUM(predicted_energy_saving_kwh) as total_predicted_saving_kwh,
                AVG(estimated_frost_thickness_mm) as avg_frost_thickness_mm,
                MAX(estimated_frost_thickness_mm) as max_frost_thickness_mm,
                AVG(confidence_score) as avg_confidence_score
            FROM defrost_optimizations
            WHERE device_id = :device_id AND timestamp >= :start_time
        """)

        stats_result = await db.execute(stats_query, {"device_id": device_id, "start_time": start_time})
        stats_row = stats_result.first()

        commands_query = text("""
            SELECT
                COUNT(*) as total_commands,
                SUM(CASE WHEN command = 'start' THEN 1 ELSE 0 END) as start_count,
                SUM(CASE WHEN command = 'stop' THEN 1 ELSE 0 END) as stop_count,
                SUM(CASE WHEN command = 'cancel' THEN 1 ELSE 0 END) as cancel_count,
                AVG(heating_power_pct) as avg_heating_power_pct,
                AVG(max_duration_minutes) as avg_duration_minutes
            FROM defrost_commands
            WHERE device_id = :device_id AND timestamp >= :start_time
        """)

        commands_result = await db.execute(commands_query, {"device_id": device_id, "start_time": start_time})
        commands_row = commands_result.first()

        status_query = text("""
            SELECT
                SUM(CASE WHEN is_defrosting THEN 1 ELSE 0 END) as defrosting_records,
                COUNT(*) as total_status_records,
                AVG(current_heating_power_pct) as avg_running_power_pct
            FROM defrost_status
            WHERE device_id = :device_id AND timestamp >= :start_time
        """)

        status_result = await db.execute(status_query, {"device_id": device_id, "start_time": start_time})
        status_row = status_result.first()

        specific_energy_kwh_per_mm = 0.5
        efficiency_coefficient = 0.85

        total_frost_removed_mm = 0
        actual_energy_saving_kwh = 0
        if stats_row and stats_row.approved_count and stats_row.avg_frost_thickness_mm:
            total_frost_removed_mm = stats_row.approved_count * stats_row.avg_frost_thickness_mm
            actual_energy_saving_kwh = total_frost_removed_mm * specific_energy_kwh_per_mm * efficiency_coefficient

        return {
            "device_id": device_id,
            "time_window_days": days,
            "optimization_stats": {
                "total_optimization_count": stats_row.total_defrost_count if stats_row else 0,
                "approved_count": stats_row.approved_count if stats_row else 0,
                "approval_rate": round(stats_row.approved_count / stats_row.total_defrost_count * 100, 2)
                if stats_row and stats_row.total_defrost_count and stats_row.total_defrost_count > 0 else 0,
                "total_predicted_saving_kwh": round(float(stats_row.total_predicted_saving_kwh), 2)
                if stats_row and stats_row.total_predicted_saving_kwh else 0,
                "actual_energy_saving_kwh": round(actual_energy_saving_kwh, 2),
                "total_frost_removed_mm": round(total_frost_removed_mm, 2),
                "avg_frost_thickness_mm": round(float(stats_row.avg_frost_thickness_mm), 2)
                if stats_row and stats_row.avg_frost_thickness_mm else 0,
                "max_frost_thickness_mm": round(float(stats_row.max_frost_thickness_mm), 2)
                if stats_row and stats_row.max_frost_thickness_mm else 0,
                "avg_confidence_score": round(float(stats_row.avg_confidence_score), 4)
                if stats_row and stats_row.avg_confidence_score else 0
            },
            "command_stats": {
                "total_commands": commands_row.total_commands if commands_row else 0,
                "start_count": commands_row.start_count if commands_row else 0,
                "stop_count": commands_row.stop_count if commands_row else 0,
                "cancel_count": commands_row.cancel_count if commands_row else 0,
                "avg_heating_power_pct": round(float(commands_row.avg_heating_power_pct), 2)
                if commands_row and commands_row.avg_heating_power_pct else 0,
                "avg_duration_minutes": round(float(commands_row.avg_duration_minutes), 2)
                if commands_row and commands_row.avg_duration_minutes else 0
            },
            "status_stats": {
                "defrosting_ratio": round(status_row.defrosting_records / status_row.total_status_records * 100, 2)
                if status_row and status_row.total_status_records and status_row.total_status_records > 0 else 0,
                "avg_running_power_pct": round(float(status_row.avg_running_power_pct), 2)
                if status_row and status_row.avg_running_power_pct else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
